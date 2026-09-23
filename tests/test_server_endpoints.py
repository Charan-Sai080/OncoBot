import unittest
import sys
import os
import asyncio
import httpx
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server import app
from onco_utils.metrics import compute_c_index, generate_km_curve_data, stratify_risk

class TestOncoBotServer(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def test_metrics(self):
        # Test C-index
        risk_scores = [1.5, 0.8, 2.2, 0.4]
        survival_times = [24, 60, 12, 72]
        censors = [1, 1, 1, 0]
        c_index = compute_c_index(risk_scores, survival_times, censors)
        self.assertGreaterEqual(c_index, 0.5)
        self.assertLessEqual(c_index, 1.0)
        
        # Test KM data
        km = generate_km_curve_data(1.45)
        self.assertIn("baseline_cohort", km)
        self.assertIn("patient_trajectory", km)
        self.assertEqual(len(km["timepoints_months"]), len(km["baseline_cohort"]))
        
        # Test risk stratification
        high_risk = stratify_risk(1.6)
        self.assertEqual(high_risk["tier"], "High Risk")
        low_risk = stratify_risk(0.6)
        self.assertEqual(low_risk["tier"], "Low Risk")

    def test_api_status(self):
        async def _test():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                res = await client.get("/api/status")
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["status"], "online")
                self.assertEqual(data["kegg_pathways_count"], 320)
        self.run_async(_test())

    def test_api_patients(self):
        async def _test():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                res = await client.get("/api/patients")
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertIn("patients", data)
                self.assertGreaterEqual(len(data["patients"]), 1)
        self.run_async(_test())

    def test_api_run_inference(self):
        async def _test():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                res = await client.post("/api/run-inference", data={"patient_id": "TCGA-2F-A9KO"})
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertTrue(data["success"])
                self.assertIn("risk_score", data)
                self.assertIn("top_pathways", data)
                self.assertGreater(len(data["top_pathways"]), 0)
        self.run_async(_test())

    def test_api_heatmap(self):
        async def _test():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                res = await client.get("/api/heatmap/TCGA-2F-A9KO")
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertIn("all_patches_coords", data)
                self.assertGreater(len(data["all_patches_coords"]), 100)
        self.run_async(_test())

    def test_api_generate_report(self):
        async def _test():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                res = await client.post("/api/generate-report", data={"patient_id": "TCGA-2F-A9KO", "risk_score": 1.428})
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertTrue(data["success"])
                self.assertIn("report_markdown", data)
                self.assertIn("EXECUTIVE SUMMARY", data["report_markdown"])
        self.run_async(_test())

    def test_index_page(self):
        async def _test():
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
                res = await client.get("/")
                self.assertEqual(res.status_code, 200)
                self.assertIn("Onco_Bot", res.text)
                self.assertTrue("<div id=\"root\">" in res.text or "genomic pathways" in res.text)
        self.run_async(_test())

if __name__ == '__main__':
    unittest.main()
