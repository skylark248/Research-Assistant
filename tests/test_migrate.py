class FakeStore:
    def __init__(self, exists=True):
        self.collection = "papers"
        self.client = self
        self._exists = exists
        self.deleted = []
        self.ensured = False
        self.pinged = False

    # VectorStore surface
    def ping(self):
        self.pinged = True

    def ensure_collection(self):
        self.ensured = True

    # client surface used by migrate
    def collection_exists(self, name):
        return self._exists

    def delete_collection(self, name):
        self.deleted.append(name)


def test_migrate_drops_and_recreates():
    from rag.migrate import migrate

    store = FakeStore(exists=True)
    migrate(store=store)
    assert store.pinged
    assert store.deleted == ["papers"]
    assert store.ensured


def test_migrate_fresh_collection_no_delete():
    from rag.migrate import migrate

    store = FakeStore(exists=False)
    migrate(store=store)
    assert store.deleted == []
    assert store.ensured
