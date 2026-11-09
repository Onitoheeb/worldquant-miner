"""
Simulator Tester Module
Handles simulation submission and monitoring
"""

import logging
import requests
import time
import threading
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class SimulationSettings:
    """Configuration for simulation parameters
    
    All alphas are risk neutralized (neutralization is never "NONE")
    """
    region: str = "USA"
    universe: str = "TOP3000"
    instrumentType: str = "EQUITY"
    delay: int = 1
    decay: int = 0
    neutralization: str = "INDUSTRY"  # Always risk neutralized, never "NONE"
    truncation: float = 0.08
    pasteurization: str = "ON"
    unitHandling: str = "VERIFY"
    nanHandling: str = "OFF"
    maxTrade: str = "OFF"
    language: str = "FASTEXPR"
    visualization: bool = False
    testPeriod: str = "P5Y0M0D"


@dataclass
class SimulationResult:
    """Result from a simulation with comprehensive data"""
    template: str
    region: str
    settings: SimulationSettings
    # Core metrics
    sharpe: float = 0.0
    fitness: float = 0.0
    turnover: float = 0.0
    returns: float = 0.0
    drawdown: float = 0.0
    margin: float = 0.0
    longCount: int = 0
    shortCount: int = 0
    # Additional metrics (from API)
    pnl: float = 0.0  # PnL data
    volatility: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    # Correlation data (stored as JSON string)
    correlations: str = ""  # JSON string of correlation data
    power_pool_corr: str = ""  # Power pool correlations
    prod_corr: str = ""  # Production correlations
    # Checks and validation
    checks: str = ""  # JSON string of checks
    # Status
    success: bool = False
    error_message: str = ""
    alpha_id: str = ""
    timestamp: float = 0.0
    # Raw data for future analysis
    raw_data: str = ""  # JSON string of full API response


class SimulatorTester:
    # Shared across instances: BRAIN enforces the concurrency limit per account.
    _submission_semaphore = threading.BoundedSemaphore(value=1)

    """
    Handles simulation submission and monitoring
    
    Separated from template generation for modularity.
    """
    
    def __init__(self, session: requests.Session, region_configs: Dict, template_generator=None):
        """
        Initialize simulator tester
        
        Args:
            session: Authenticated requests session
            region_configs: Region configuration dictionary
            template_generator: Optional reference to template generator for re-authentication
        """
        self.sess = session
        self.region_configs = region_configs
        self.template_generator = template_generator  # For re-authentication if needed
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.active_simulations = {}  # {alpha_id: Future}
        # BRAIN's concurrency limit is account-wide, not per GUI worker or tester.
        # A class-level semaphore serializes submissions across every SimulatorTester
        # instance created by Step 5 workers.

    
    def submit_simulation(
        self, 
        template: str, 
        region: str, 
        settings: SimulationSettings
    ) -> Optional[str]:
        """
        Submit a template for simulation
        
        Args:
            template: Alpha expression
            region: Region code
            settings: Simulation settings
            
        Returns:
            Progress URL if successful, None otherwise
        """
        permit_acquired = False
        try:
            import re
            from dataclasses import asdict

            # Final safety barrier: never send unresolved algorithmic placeholders to BRAIN.
            unresolved = re.findall(r'\b(?:OPERATOR|DATA_FIELD)\d+\b', template or '', re.IGNORECASE)
            if unresolved:
                logger.error(
                    "Rejecting simulation submission because unresolved placeholders remain: %s",
                    unresolved
                )
                return None

            # Require compiler validation at the API boundary as well. This protects
            # retries and any caller that bypasses the Step 5 GUI validation path.
            validator = getattr(self.template_generator, 'template_validator', None)
            if validator and getattr(validator, 'use_ast', False):
                compile_result = validator.compile_template(template, optimize=False)
                if not compile_result.success:
                    errors = [getattr(error, 'message', str(error)) for error in (compile_result.errors or [])]
                    logger.error(
                        "Rejecting simulation submission because FASTEXPR compilation failed: %s",
                        errors[:3]
                    )
                    return None
                if compile_result.final_expression:
                    template = compile_result.final_expression
            
            region_config = self.region_configs.get(region)
            if not region_config:
                logger.error(f"Unknown region: {region}")
                return None
            
            # Verify session has cookies before making request (for debugging)
            if not self.sess.cookies:
                logger.warning("⚠️ Session has no cookies - authentication may have expired")
                # Try to re-authenticate if template_generator is available
                if self.template_generator:
                    logger.info("Attempting to re-authenticate...")
                    self.template_generator.setup_auth()
            
            # Prepare simulation data in the correct format
            # Update settings with region-specific values
            settings_dict = asdict(settings)
            settings_dict['region'] = region
            settings_dict['universe'] = region_config.universe
            settings_dict['instrumentType'] = settings.instrumentType
            
            # Ensure neutralization is always risk neutralized (never "NONE")
            # Use region-specific default if neutralization is NONE or empty
            if settings_dict.get('neutralization') == 'NONE' or not settings_dict.get('neutralization'):
                try:
                    from .region_config import get_default_neutralization
                    settings_dict['neutralization'] = get_default_neutralization(region)
                    logger.info(f"⚠️ Neutralization was NONE or empty, using default risk neutralization: {settings_dict['neutralization']}")
                except ImportError:
                    # Try to get from region_config if available
                    if hasattr(region_config, 'neutralization'):
                        settings_dict['neutralization'] = region_config.neutralization
                    else:
                        settings_dict['neutralization'] = 'INDUSTRY'  # Fallback
                    logger.warning(f"⚠️ Neutralization was NONE or empty, using fallback: {settings_dict['neutralization']}")
            
            # Final check: ensure neutralization is never "NONE"
            if settings_dict.get('neutralization') == 'NONE':
                settings_dict['neutralization'] = 'INDUSTRY'
                logger.warning(f"⚠️ Forced neutralization from NONE to INDUSTRY (all alphas must be risk neutralized)")
            
            simulation_data = {
                'type': 'REGULAR',
                'settings': settings_dict,
                'regular': template
            }
            
            # Serialize submissions and keep the permit until the corresponding
            # monitor_simulation call finishes. This prevents concurrent POST bursts.
            type(self)._submission_semaphore.acquire()
            permit_acquired = True
            # Retry only explicit concurrency-limit rejections. A 429 response
            # means this POST was rejected, so retrying cannot create a duplicate alpha.
            response = None
            for attempt in range(6):
                if self.template_generator and hasattr(self.template_generator, 'make_api_request'):
                    response = self.template_generator.make_api_request(
                        'POST',
                        'https://api.worldquantbrain.com/simulations',
                        json=simulation_data
                    )
                else:
                    response = self.sess.post(
                        'https://api.worldquantbrain.com/simulations',
                        json=simulation_data
                    )
                if response.status_code != 429 or "CONCURRENT_SIMULATION_LIMIT_EXCEEDED" not in (response.text or ""):
                    break
                if attempt < 5:
                    wait_seconds = min(30, 5 * (attempt + 1))
                    logger.warning(
                        "BRAIN concurrent simulation limit reached; retrying in %ss (%s/5)",
                        wait_seconds, attempt + 1
                    )
                    time.sleep(wait_seconds)

            if response.status_code == 201:
                progress_url = response.headers.get('Location')
                if not progress_url:
                    logger.error("No Location header in response")
                    type(self)._submission_semaphore.release()
                    permit_acquired = False
                    return None
                logger.info(f"Submitted simulation: {progress_url} for region {region}")
                # The monitor wrapper now owns responsibility for releasing it.
                permit_acquired = False
                return progress_url
            elif response.status_code == 401:
                logger.error(f"Authentication expired - session cookies may have been lost")
                logger.error(f"Response: {response.text}")
                self._submission_semaphore.release()
                permit_acquired = False
                return None
            else:
                logger.error(f"Failed to submit simulation: {response.status_code} - {response.text}")
                self._submission_semaphore.release()
                permit_acquired = False
                return None
                
        except Exception as e:
            logger.error(f"Error submitting simulation: {e}")
            # A failed POST never transfers the permit to a monitor.
            if permit_acquired:
                self._submission_semaphore.release()
            return None
    
    def monitor_simulation(
        self,
        progress_url: str,
        template: str,
        region: str,
        settings: SimulationSettings,
        max_wait_time: int = 300,
        progress_callback: Optional[Callable[[float, str, str], None]] = None
    ) -> SimulationResult:
        """Monitor a submitted simulation and always release its concurrency permit."""
        try:
            return self._monitor_simulation_impl(
                progress_url, template, region, settings,
                max_wait_time=max_wait_time,
                progress_callback=progress_callback
            )
        finally:
            try:
                self._submission_semaphore.release()
            except ValueError:
                logger.warning("Simulation concurrency permit was already released")

    def _monitor_simulation_impl(
        self,
        progress_url: str,
        template: str,
        region: str,
        settings: SimulationSettings,
        max_wait_time: int = 300,
        progress_callback: Optional[Callable[[float, str, str], None]] = None
    ) -> SimulationResult:
        """
        Monitor a simulation until completion
        
        Args:
            progress_url: Progress URL from submission
            template: Original template
            region: Region code
            settings: Simulation settings
            max_wait_time: Maximum wait time in seconds
            progress_callback: Optional callback(progress_percent, message, api_status) for progress updates
            
        Returns:
            SimulationResult
        """
        start_time = time.time()
        alpha_id = ""
        last_status = ""
        last_progress_update = 0
        
        while time.time() - start_time < max_wait_time:
            try:
                # Use make_api_request if available for better session persistence
                if self.template_generator and hasattr(self.template_generator, 'make_api_request'):
                    response = self.template_generator.make_api_request(
                        'GET',
                        progress_url
                    )
                else:
                    response = self.sess.get(progress_url)
                
                if response.status_code == 401:
                    logger.warning("Session expired, need to re-authenticate")
                    # Try to re-authenticate if template_generator is available
                    if self.template_generator:
                        logger.info("Attempting to re-authenticate...")
                        self.template_generator.setup_auth()
                        # Retry the request
                        if hasattr(self.template_generator, 'make_api_request'):
                            response = self.template_generator.make_api_request(
                                'GET',
                                progress_url
                            )
                        else:
                            response = self.sess.get(progress_url)
                    
                    if response.status_code == 401:
                        return SimulationResult(
                            template=template,
                            region=region,
                            settings=settings,
                            success=False,
                            error_message="Authentication expired",
                            alpha_id=alpha_id,
                            timestamp=time.time()
                        )
                
                if response.status_code != 200:
                    # Calculate progress based on elapsed time (rough estimate)
                    elapsed = time.time() - start_time
                    estimated_progress = min(90.0, (elapsed / max_wait_time) * 100)
                    if progress_callback:
                        progress_callback(estimated_progress, f"Waiting for response... (HTTP {response.status_code})", "WAITING")
                    time.sleep(5)  # Reduced from 10s to 5s for more frequent updates
                    continue
                
                data = response.json()
                status = str(data.get('status', 'UNKNOWN')).upper()

                # BRAIN commonly reports progress as a fraction and signals completion
                # by returning an alpha ID, rather than a textual status field.
                elapsed = time.time() - start_time
                raw_progress = data.get('progress')
                api_progress = None
                if isinstance(raw_progress, (int, float)):
                    api_progress = float(raw_progress)
                    if 0.0 <= api_progress <= 1.0:
                        api_progress *= 100.0
                    api_progress = max(0.0, min(100.0, api_progress))

                if data.get('alpha'):
                    status = 'COMPLETE'
                elif status in ('UNKNOWN', ''):
                    if api_progress is not None:
                        status = 'RUNNING' if api_progress > 0 else 'PENDING'
                        logger.debug(
                            f"Simulation API progress={raw_progress!r}; interpreted status={status}"
                        )
                    elif elapsed < 10:
                        status = 'PENDING'
                        logger.debug(
                            f"Status missing in early simulation response; treating as PENDING (elapsed: {elapsed:.1f}s)"
                        )
                    else:
                        logger.warning(f"Simulation response has no status/progress/alpha: {data}")
                
                # Calculate progress based on status and elapsed time
                if status == 'COMPLETE':
                    progress_percent = 100.0
                    progress_msg = "✅ Simulation complete"
                elif status in ['FAILED', 'ERROR']:
                    progress_percent = 100.0
                    progress_msg = f"❌ Simulation failed: {data.get('message', 'Unknown error')[:50]}"
                elif status == 'PENDING' or status == 'CREATED' or status == 'SUBMITTED' or status == 'QUEUED':
                    # Treat various "just submitted" statuses as PENDING
                    progress_percent = min(15.0, (elapsed / max_wait_time) * 15)
                    progress_msg = f"⏳ Simulation pending... ({int(elapsed)}s)"
                elif status == 'RUNNING' or status == 'IN_PROGRESS' or status == 'PROCESSING':
                    # Prefer the API's actual progress value when present; otherwise estimate.
                    if api_progress is not None:
                        progress_percent = min(99.0, api_progress)
                    else:
                        base_progress = 15.0
                        time_based_progress = min(85.0, (elapsed / max_wait_time) * 70)
                        progress_percent = min(95.0, base_progress + time_based_progress)
                    progress_msg = f"🔄 Simulation running... {progress_percent:.0f}% ({int(elapsed)}s elapsed)"
                else:
                    # Unknown status, estimate based on time
                    # Log it for debugging but treat as PENDING if early, RUNNING if later
                    if elapsed < 30:
                        # Early stage - treat as PENDING
                        progress_percent = min(15.0, (elapsed / max_wait_time) * 15)
                        progress_msg = f"⏳ Simulation initializing... ({int(elapsed)}s)"
                        logger.debug(f"Unknown status '{status}' treated as PENDING (early stage)")
                    else:
                        # Later stage - treat as RUNNING
                        base_progress = 15.0
                        time_based_progress = min(85.0, (elapsed / max_wait_time) * 70)
                        progress_percent = min(95.0, base_progress + time_based_progress)
                        progress_msg = f"🔄 Simulation in progress... ({int(elapsed)}s)"
                        logger.debug(f"Unknown status '{status}' treated as RUNNING (later stage)")
                
                # Call progress callback if provided (throttle to every 2 seconds to avoid spam)
                current_time = time.time()
                if progress_callback and (current_time - last_progress_update >= 2.0 or status != last_status):
                    progress_callback(progress_percent, progress_msg, status)
                    last_progress_update = current_time
                
                # Update last_status for next iteration
                last_status = status
                
                if status == 'COMPLETE':
                    # Get alpha ID from the simulation response
                    alpha_id_raw = data.get('alpha', '')
                    # Ensure alpha_id is a string, not a tuple or list
                    if isinstance(alpha_id_raw, (tuple, list)):
                        alpha_id = str(alpha_id_raw[0]) if alpha_id_raw else ''
                    else:
                        alpha_id = str(alpha_id_raw) if alpha_id_raw else ''
                    if not alpha_id:
                        logger.error("No alpha ID in completed simulation response")
                        return SimulationResult(
                            template=template,
                            region=region,
                            settings=settings,
                            success=False,
                            error_message="No alpha ID in response",
                            alpha_id="",
                            timestamp=time.time()
                        )
                    
                    # Do not record a successful alpha-ID extraction as a simulation error.
                    # The previous implementation incorrectly taught the validator that every
                    # completed simulation had "No alpha ID in response" before fetching alpha details.

                    # Get alpha details from the alpha endpoint
                    if self.template_generator and hasattr(self.template_generator, 'make_api_request'):
                        alpha_response = self.template_generator.make_api_request(
                            'GET',
                            f'https://api.worldquantbrain.com/alphas/{alpha_id}'
                        )
                    else:
                        alpha_response = self.sess.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}')
                    
                    if alpha_response.status_code == 200:
                        import json
                        alpha_data = alpha_response.json()
                        is_data = alpha_data.get('is', {})
                        
                        # Extract all available data
                        correlations_data = alpha_data.get('correlations', {})
                        power_pool_corr = correlations_data.get('powerPool', {})
                        prod_corr = correlations_data.get('prod', {})
                        checks_data = is_data.get('checks', [])
                        
                        # Calculate additional metrics
                        pnl = is_data.get('pnl', is_data.get('returns', 0.0))  # PnL or returns
                        volatility = is_data.get('volatility', 0.0)
                        max_drawdown = is_data.get('maxDrawdown', is_data.get('drawdown', 0.0))
                        
                        # Store raw data as JSON for future analysis
                        raw_data_json = json.dumps(alpha_data)
                        correlations_json = json.dumps(correlations_data) if correlations_data else ""
                        power_pool_json = json.dumps(power_pool_corr) if power_pool_corr else ""
                        prod_corr_json = json.dumps(prod_corr) if prod_corr else ""
                        checks_json = json.dumps(checks_data) if checks_data else ""
                        
                        # Check for warnings (v2 style: mark warnings as red if no red errors)
                        has_warnings = self._has_warnings_only(checks_data)
                        # If there are warnings but no red errors, mark as failed (red) like v2
                        is_success = not has_warnings
                        warning_message = ""
                        if has_warnings:
                            warning_message = "Alpha has warnings (marked as red per v2 behavior)"
                        
                        return SimulationResult(
                            template=template,
                            region=region,
                            settings=settings,
                            sharpe=is_data.get('sharpe', 0.0),
                            fitness=is_data.get('fitness', 0.0),
                            turnover=is_data.get('turnover', 0.0),
                            returns=is_data.get('returns', 0.0),
                            drawdown=is_data.get('drawdown', 0.0),
                            margin=is_data.get('margin', 0.0),
                            longCount=is_data.get('longCount', 0),
                            shortCount=is_data.get('shortCount', 0),
                            pnl=pnl,
                            volatility=volatility,
                            max_drawdown=max_drawdown,
                            win_rate=is_data.get('winRate', 0.0),
                            avg_return=is_data.get('avgReturn', 0.0),
                            correlations=correlations_json,
                            power_pool_corr=power_pool_json,
                            prod_corr=prod_corr_json,
                            checks=checks_json,
                            success=is_success,
                            error_message=warning_message if has_warnings else "",
                            alpha_id=alpha_id,
                            timestamp=time.time(),
                            raw_data=raw_data_json
                        )
                    else:
                        error_msg = f"Failed to get alpha details: {alpha_response.status_code}"
                        
                        # Learn from error
                        if self.template_generator and hasattr(self.template_generator, 'template_validator'):
                            validator = self.template_generator.template_validator
                            if validator:
                                validator.learn_from_simulation_error(template, error_msg)
                        
                        return SimulationResult(
                            template=template,
                            region=region,
                            settings=settings,
                            success=False,
                            error_message=error_msg,
                            alpha_id=alpha_id,
                            timestamp=time.time()
                        )
                elif status in ['FAILED', 'ERROR']:
                    error_msg = data.get('message', 'Unknown error')
                    
                    # Learn from simulation error
                    if self.template_generator and hasattr(self.template_generator, 'template_validator'):
                        validator = self.template_generator.template_validator
                        if validator:
                            validator.learn_from_simulation_error(template, error_msg)
                    
                    return SimulationResult(
                        template=template,
                        region=region,
                        settings=settings,
                        success=False,
                        error_message=error_msg,
                        alpha_id=alpha_id,
                        timestamp=time.time()
                    )
                
                # Wait before checking again (shorter wait for better progress updates)
                time.sleep(5)  # Check every 5 seconds for better progress tracking
                
            except Exception as e:
                logger.error(f"Error monitoring simulation {progress_url}: {e}")
                time.sleep(10)
        
        # Timeout
        return SimulationResult(
            template=template,
            region=region,
            settings=settings,
            success=False,
            error_message="Simulation timeout",
            alpha_id=alpha_id,
            timestamp=time.time()
        )
    
    def _has_warnings_only(self, checks_data: List) -> bool:
        """
        Check if alpha has warnings but no red errors (v2 behavior: mark as red)
        
        Args:
            checks_data: List of check objects from API
            
        Returns:
            True if there are warnings but no red errors
        """
        if not checks_data:
            return False
        
        has_red_error = False
        has_warning = False
        
        for check in checks_data:
            if isinstance(check, dict):
                # Check for red errors (status: 'ERROR' or 'FAILED' or similar)
                status = check.get('status', '').upper()
                check_type = check.get('type', '').upper()
                severity = check.get('severity', '').upper()
                
                # Red errors typically have status ERROR, FAILED, or severity ERROR
                if status in ['ERROR', 'FAILED', 'RED'] or severity == 'ERROR':
                    has_red_error = True
                    break
                
                # Warnings typically have status WARNING, severity WARNING, or type WARNING
                if status == 'WARNING' or severity == 'WARNING' or check_type == 'WARNING':
                    has_warning = True
            elif isinstance(check, str):
                # Handle string format checks
                check_upper = check.upper()
                if 'ERROR' in check_upper or 'FAILED' in check_upper or 'RED' in check_upper:
                    has_red_error = True
                    break
                if 'WARNING' in check_upper:
                    has_warning = True
        
        # Return True if there are warnings but no red errors (v2 behavior)
        return has_warning and not has_red_error
    
    def simulate_template_concurrent(
        self, 
        template: str, 
        region: str, 
        settings: SimulationSettings
    ) -> Future:
        """
        Submit and monitor simulation concurrently
        
        Args:
            template: Alpha expression
            region: Region code
            settings: Simulation settings
            
        Returns:
            Future object for the simulation
        """
        def run_simulation():
            progress_url = self.submit_simulation(template, region, settings)
            if not progress_url:
                return SimulationResult(
                    template=template,
                    region=region,
                    settings=settings,
                    success=False,
                    error_message="Failed to submit",
                    timestamp=time.time()
                )
            
            return self.monitor_simulation(progress_url, template, region, settings)
        
        future = self.executor.submit(run_simulation)
        return future
    
    def simulate_batch(
        self, 
        templates: List[str], 
        region: str, 
        settings: SimulationSettings
    ) -> List[Future]:
        """
        Submit multiple simulations concurrently
        
        Args:
            templates: List of alpha expressions
            region: Region code
            settings: Simulation settings
            
        Returns:
            List of Future objects
        """
        futures = []
        for template in templates:
            future = self.simulate_template_concurrent(template, region, settings)
            futures.append(future)
            time.sleep(0.5)  # Rate limiting
        
        logger.info(f"Submitted {len(futures)} simulations for region {region}")
        return futures
    
    def wait_for_results(self, futures: List[Future], timeout: int = 600) -> List[SimulationResult]:
        """
        Wait for simulation results
        
        Args:
            futures: List of Future objects
            timeout: Maximum wait time
            
        Returns:
            List of SimulationResult objects
        """
        results = []
        start_time = time.time()
        
        for future in futures:
            if time.time() - start_time > timeout:
                logger.warning("Timeout waiting for results")
                break
            
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                logger.error(f"Error getting result: {e}")
                results.append(SimulationResult(
                    template="",
                    region="",
                    settings=SimulationSettings(),
                    success=False,
                    error_message=str(e),
                    timestamp=time.time()
                ))
        
        return results

