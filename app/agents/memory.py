class AgentMemory:
    def __init__(self):
        self.steps = []

    def add_steps(self, thought, action, results):
        self.steps.append({
            "thought": thought,
            "action": action,
            "result": str(results)[:500]
        })

    def get_context(self):
        return "\n".join([
            f"Step {i+1}:\nThought: {s['thought']}\nAction: {s['action']}\nResult: {s['result']}"
            for i, s in enumerate(self.steps)
        ])
            