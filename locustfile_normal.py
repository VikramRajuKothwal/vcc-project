# locustfile_normal.py
from locust import HttpUser, task, between

class NormalUser(HttpUser):
    wait_time = between(1, 3)

    @task(5)
    def visit_homepage(self):
        self.client.get("/")

    @task(2)
    def visit_slow(self):
        self.client.get("/", timeout=10)

    @task(1)
    def visit_pause(self):
        self.client.get("/")
