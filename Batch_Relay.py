class BatchQueue:
    def __init__(self):
        self.length = 0
        self.head = None
        self.last = None
    def is_empty(self):
        return self.is_empty.length == 0
    def insert(self, img):
        last = self.last
        last.next = img
        self.last = img
        self.length += 1
    def remove(self):
        self.head = self.head.next
        self.length -= 1
    batch = BatchQueue()