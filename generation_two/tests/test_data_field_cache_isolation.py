from generation_two.data_fetcher.data_field_fetcher import DataFieldFetcher


def test_memory_cache_is_scoped_to_region_delay_and_universe(tmp_path):
    fetcher = DataFieldFetcher(session=None, cache_dir=str(tmp_path))
    usa_top3000 = [{"id": "close", "region": "USA", "universe": "TOP3000", "delay": 1}]
    usa_top1000 = [{"id": "close", "region": "USA", "universe": "TOP1000", "delay": 1}]
    usa_top3000_delay0 = [{"id": "close", "region": "USA", "universe": "TOP3000", "delay": 0}]

    fetcher._memory_cache[("USA", 1, "TOP3000")] = usa_top3000
    fetcher._memory_cache[("USA", 1, "TOP1000")] = usa_top1000
    fetcher._memory_cache[("USA", 0, "TOP3000")] = usa_top3000_delay0

    assert fetcher._fetch_data_fields_for_universe("USA", 1, "TOP3000") == usa_top3000
    assert fetcher._fetch_data_fields_for_universe("USA", 1, "TOP1000") == usa_top1000
    assert fetcher._fetch_data_fields_for_universe("USA", 0, "TOP3000") == usa_top3000_delay0


def test_memory_cache_miss_does_not_return_other_universe(tmp_path):
    fetcher = DataFieldFetcher(session=None, cache_dir=str(tmp_path))
    fetcher._memory_cache[("USA", 1, "TOP3000")] = [
        {"id": "close", "region": "USA", "universe": "TOP3000", "delay": 1}
    ]

    # With no API session and no exact-key cache, the fetch must fail closed
    # instead of leaking the TOP3000 fields into the TOP1000 request.
    assert fetcher._fetch_data_fields_for_universe("USA", 1, "TOP1000") == []
