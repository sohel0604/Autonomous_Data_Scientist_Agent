# agents/coordinator.py
from agents.agents_impl import DataLoaderAgent, EDAAgent, ModelTrainerAgent, ReportGeneratorAgent

class Coordinator:
    def __init__(self):
        self.loader = DataLoaderAgent()
        self.eda_agent = EDAAgent()
        self.trainer = ModelTrainerAgent()
        self.reporter = ReportGeneratorAgent()  # no API key required

    def run_pipeline(self, context):
        context = self.loader.run(context)
        context = self.eda_agent.run(context)
        context = self.trainer.run(context)
        context = self.reporter.run(context)
        return context
