import re
import pandas as pd

def parse_logs_to_dataframe(log_text):
    # Regex patterns to extract the necessary values
    pattern = re.compile(
        r"Question (\d+)/.*?Ground Truth: (.*?)\n.*?"
        r"Original probs: ([\d\.]+).*?"
        r"baseline_del probs: ([\d\.]+).*?"
        r"baseline_ins probs: ([\d\.]+).*?"
        r"Prob when Inserting Mask \(top 20%\): ([\d\.]+).*?"
        r"Prob when Deleting Mask \(top 20%\): ([\d\.]+)",
        re.DOTALL
    )
    
    matches = pattern.findall(log_text)
    data = []

    for match in matches:
        q_num = match[0]
        gt = match[1].strip()
        orig_prob = float(match[2])
        base_del = float(match[3])
        base_ins = float(match[4])
        prob_ins = float(match[5])
        prob_del = float(match[6])

        orig_diff_ins = orig_prob - base_ins # Same for this
        orig_diff_del = orig_prob - base_del # We expect original probabilities to be higher
        diff_ins = prob_ins - base_ins
        diff_del = prob_del - base_del
    
        # Calculate Relative Insertion
        # Protect against division by zero if orig_prob == base_ins
        if orig_diff_ins != 0:
            rel_ins = (prob_ins - base_ins) / orig_diff_ins
        else:
            rel_ins = 0.0

        if orig_diff_del != 0:
            rel_del = (orig_prob - prob_del) / orig_diff_del
        else:
            rel_del = 0.0

        data.append({
            "Question": q_num,
            "Ground Truth": gt,
            "Orig Prob": orig_prob,
            "Base Ins": base_ins,
            "Ins Prob": prob_ins,
            "Orig Ins diff": orig_diff_ins,
            "Mask insertion diff": diff_ins,
            "Rel Ins (%)": round(rel_ins * 100, 2),
            "Orig Prob.": orig_prob,
            "Base Del": base_del,
            "Del Prob": prob_del,
            "Orig Del diff": orig_diff_del,
            "Mask deletion diff": diff_del,
            "Rel Del (%)": round(rel_del * 100, 2)
        })

    return pd.DataFrame(data)

# Usage Example:
with open("/home/s2498278/results/spix_imagenet/424-119/log.txt", "r") as file:
    log_data = file.read()

print("Data read")

df = parse_logs_to_dataframe(log_data)
print(df.to_string())
df.to_csv("/home/s2498278/results/spix_imagenet/424-119/log.csv", index=False)