from data_sync import DataSyncService, FlakyDataSource


def test_data_sync_service_can_be_constructed():
    service = DataSyncService(FlakyDataSource(fail_times=0))
    assert service.source.fail_times == 0
