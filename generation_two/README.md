# Generation Two: Self-Optimizing Alpha Mining System

Generation Two is a self-optimizing, modular alpha mining system designed for WorldQuant Brain. It extends Generation One with advanced capabilities including self-optimization, genetic algorithm-based evolution, continuous mining, template validation with error learning, and a modern cyberpunk GUI interface.

## 🚀 Quick Start

```bash
cd generation_two
pip install -r requirements.txt
python gui/run_gui.py [credential_path]
```

## 📖 Documentation

**Complete documentation is available in [DOCUMENTATION.md](DOCUMENTATION.md)**

The consolidated documentation includes:
- Overview and architecture
- Core components and features
- Installation and usage
- Configuration
- Building and release
- Troubleshooting

## ✨ Key Features

- **Self-Optimization**: Adaptive parameter tuning based on performance
- **Genetic Evolution**: Genetic algorithm-based alpha evolution
- **Continuous Mining**: Automated 24/7 alpha discovery with error correction
- **Cyberpunk GUI**: Modern graphical interface for system control
- **Smart Ollama Integration**: Local LLM with automatic fallback
- **Template Validation**: Strict FASTEXPR AST/compiler validation with evidence-based error learning
- **Autonomous Research Memory**: Persistent SQLite memory of BRAIN experiments, hypotheses, operator/field catalogues, correlations, and failure lessons
- **LLM-led Alpha Research**: Step 4 proposes complete expressions from measured research seeds and current BRAIN metadata, then requires local compilation before simulation
- **Expression Compiler**: Multi-stage compilation pipeline
- **Quality Monitoring**: Performance tracking and degradation detection

## 🧠 Research Feedback Loop

Generation Two now maintains research memory in the configured backtest SQLite database.

1. Step 4 snapshots the available BRAIN operators and data fields, including their metadata.
2. When Ollama is available, the research path asks the LLM to propose a complete expression using exact catalogue names and historical experiment context.
3. A candidate must pass local FASTEXPR compilation before it is queued for Step 5.
4. Step 5 records the observed BRAIN simulation outcome, including performance metrics, checks, and available correlation data.
5. Later research prompts retrieve diverse historical candidates and recent failure lessons.

This is persistent retrieval-based learning, not automatic model-weight retraining. A locally compiled expression is not considered proven until BRAIN simulation supplies the evidence. The existing algorithmic placeholder generator remains the fallback when Ollama is unavailable or its candidate fails compilation.

## 🏗️ Architecture

Modular architecture with separated concerns:
- **Core**: Template generation, simulation testing, validation
- **Evolution**: Self-optimization, genetic algorithms, quality monitoring
- **Storage**: Backtest storage, regrouping, retrospective analysis
- **Ollama**: Smart Ollama integration with fallback
- **GUI**: Cyberpunk-themed graphical interface
- **Mining**: Continuous mining coordination

## 📦 Building

See [BUILD.md](BUILD.md) for build instructions.

**Windows (EXE):**
```bash
python build.py --exe
```

**Linux (DEB):**
```bash
python build.py --deb
```

**macOS (DMG):**
```bash
python build.py --dmg
```

## 🔒 Security

- Credentials never embedded in code
- All credentials loaded from external files or user input
- Security verification script: `python verify_secrets.py`

## 📚 Additional Resources

- **Specification**: `spec/main.tex` - Complete LaTeX specification
- **Architecture**: `spec/architecture.tex` - Architecture documentation
- **Expression Compiler**: `spec/expression_compiler.tex` - Compiler specification
- **Smart Search**: `spec/smart_search.tex` - Search engine specification

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
