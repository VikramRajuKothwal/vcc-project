#locustfile_stress.py
from locust import HttpUser, task, between

class StressUser(HttpUser):
    wait_time = between(0.1, 0.5)  # very fast requests

    @task(10)
    def hammer_homepage(self):
        self.client.get("/")

    @task(5)
    def concurrent_slow(self):
        self.client.get("/", timeout=30)

    @task(3)
    def rapid_fire(self):
        for _ in range(5):
            self.client.get("/")
