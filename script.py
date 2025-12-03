def helper(text: str):
    for line in text.split("\n"):
        data = line.strip().split("&")
        tr_delta = float(data[0]) - float(data[2])
        test_delta = float(data[1]) - float(data[3])
        print(f"& {tr_delta:.3f} & {test_delta:.3f}")

if __name__ == "__main__":
    text = " 0.047 & 0.157 & 0.011 & 0.195"
    helper(text)