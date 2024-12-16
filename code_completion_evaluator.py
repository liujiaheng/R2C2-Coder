import json
from transformers import AutoModelForCausalLM, AutoTokenizer


class CodeCompletionEvaluator:
    def __init__(self, model_name="bigcode/starcoder"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16,
                                                          device_map="auto").eval()
        print('Model and tokenizer are loaded.')
        self.metrics = {
            'python': {'exact_match': 0, 'edit_sim': 0, 'count': 0},
            'java': {'exact_match': 0, 'edit_sim': 0, 'count': 0},
            'typescript': {'exact_match': 0, 'edit_sim': 0, 'count': 0},
            'csharp': {'exact_match': 0, 'edit_sim': 0, 'count': 0}
        }

    def levenshtein_distance(self, s1, s2):
        m = len(s1)
        n = len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

        return dp[m][n]

    def edit_similarity(self, text1, text2):
        distance = self.levenshtein_distance(text1, text2)
        similarity = 1 - (distance / max(len(text1), len(text2)))
        return similarity

    def evaluate(self, jsonl_file):
        with open(jsonl_file, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                language = data['language'].lower()
                prefix = data['prefix']
                suffix = data['suffix']
                middle = data['middle']

                input_text = '<｜fim▁begin｜>' + prefix + '<｜fim▁hole｜>' + suffix + '<｜fim▁end｜>'
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to("cuda")
                output_ids = self.model.generate(input_ids, max_new_tokens=256, do_sample=True, top_k=1)
                prediction = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).split('<｜fim▁end｜>')[-1]

                exact_match = 1 if prediction == middle else 0
                edit_sim = self.edit_similarity(prediction, middle)

                self.metrics[language]['exact_match'] += exact_match
                self.metrics[language]['edit_sim'] += edit_sim
                self.metrics[language]['count'] += 1

    def report_metrics(self):
        for language, stats in self.metrics.items():
            count = stats['count']
            exact_match_rate = stats['exact_match'] / count if count > 0 else 0
            edit_sim_rate = stats['edit_sim'] / count if count > 0 else 0
            print(f"{language}: Exact Match Rate = {exact_match_rate:.4f}, Edit Similarity Rate = {edit_sim_rate:.4f}")


if __name__ == "__main__":
    evaluator = CodeCompletionEvaluator(model_name='deepseek-ai/deepseek-coder-1.3b-base')
    evaluator.evaluate("r2c2.jsonl")
    # evaluator.evaluate("m2rc_eval.jsonl")
    evaluator.report_metrics()
