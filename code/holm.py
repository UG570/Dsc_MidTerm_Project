from stat_helpers import holm, load_json, save_json

def main():
    ps1, ps2 = load_json("ps1.json"), load_json("ps2.json")
    p = {
        "Test 1": ps1["test1"]["p_value"],
        "Test 2": ps1["test2"]["p_value"],
        "Test 3": ps2["test3"]["p_value"],
        "Test 4": ps2["test4"]["p_value"],
    }
    res = holm(p)
    for name in sorted(res):
        r = res[name]
        raw_reject = r["p_raw"] <= 0.05
        r["reject_raw"] = raw_reject
        r["decision_changed"] = raw_reject != r["reject_holm"]
        print(f"{name}: p={r['p_raw']:.3g} Holm p={r['p_holm']:.3g}"
              f" reject raw={raw_reject} Holm={r['reject_holm']}")
    save_json(res, "holm.json")


if __name__ == "__main__":
    main()
