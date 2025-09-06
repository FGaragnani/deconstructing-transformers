from typing import List, Optional
from scipy.stats import mannwhitneyu
import re

FILE_NAME = "results_log.txt"


class Result:
    def __init__(self, id: Optional[int] = None, category: Optional[str] = None, train_t: Optional[float] = None, train_f: Optional[float] = None, test_t: Optional[float] = None, test_f: Optional[float] = None):
        self.id = id
        self.category = category
        self.train_t = train_t
        self.train_f = train_f
        self.test_t = test_t
        self.test_f = test_f

if __name__ == "__main__":
    
    with open(FILE_NAME, "r+") as file:
        lines: List[str] = file.readlines()
    
    results: List[Result] = []
    current_result = Result()
    for line in lines:
        if line.strip() == "":
            if current_result.category is not None or current_result.id is not None:
                results.append(current_result)
            current_result = Result()
            continue

        if line.startswith("Dataset:"):
            parts = line.split()
            if len(parts) >= 2:
                current_result.category = parts[1]
            m = re.search(r"\(ID:\s*(\d+)\)", line)
            if m:
                current_result.id = int(m.group(1))
            continue

        if line.startswith("Transformer - "):
            nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
            if len(nums) >= 2:
                current_result.train_t = float(nums[0])
                current_result.test_t = float(nums[1])
            continue

        if line.startswith("Random Forest - "):
            nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
            if len(nums) >= 2:
                current_result.train_f = float(nums[0])
                current_result.test_f = float(nums[1])
            continue
    
    categories: List[str] = list(set([result.category for result in results if result.category is not None]))
    
    train_arr_t: List[float] = []
    test_arr_t: List[float] = []
    train_arr_f: List[float] = []
    test_arr_f: List[float] = []
    
    for category in categories:
        part_results = [result for result in results if result.category == category]
        train_total_tr = sum([result.train_t if result.train_t is not None else 0 for result in part_results]) / len(part_results)
        test_total_tr = sum([result.test_t if result.test_t is not None else 0 for result in part_results]) / len(part_results)
        train_total_rf = sum([result.train_f if result.train_f is not None else 0 for result in part_results]) / len(part_results)
        test_total_rf = sum([result.test_f if result.test_f is not None else 0 for result in part_results]) / len(part_results)

        train_arr_t.append(train_total_tr)
        train_arr_f.append(train_total_rf)
        test_arr_t.append(test_total_tr)
        test_arr_f.append(test_total_rf)
    
        print(f"""
            CATEGORY {category}
            -------------------
            Train Tot: 
                Transformer     - {train_total_tr:4f}
                Random Forest   - {train_total_rf:4f}
            Test Tot:
                Transformer     - {test_total_tr:4f}
                Random Forest   - {test_total_rf:4f}

            Total Series: {len(part_results)}
            
        """)
    p_val_test = mannwhitneyu(test_arr_t, test_arr_f, alternative="two-sided").pvalue
    p_val_train = mannwhitneyu(train_arr_t, train_arr_f, alternative="two-sided").pvalue
    print(f" Test p-value: {p_val_test}")
    print(f" Train p-value: {p_val_train}")