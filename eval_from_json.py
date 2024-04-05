import numpy as np


from eval_utils import _evaluate


def main():
    if True:
        root_output_path = "checkpoints/flow_v3_Quasar_T_a100/debug0_eulerstepsize0.001_can15_anchor0_samples20230622-00:41:12"
        _evaluate(
            _folder=root_output_path,
            _mbr=True,
            _eos="[SEP]",
            _sos="[CLS]",
            _sep="[SEP]",
            _pad="[PAD]",
            candidate_num=-1,
        )
    elif True:
        with open(f"{root_output_path}_eval.txt", "w") as f:
            f.write(f"hello kitty\n")
            f.flush()
            for _c in range(1, 16):
                _dict = _evaluate(
                    _folder=root_output_path,
                    _mbr=True,
                    _eos="[SEP]",
                    _sos="[CLS]",
                    _sep="[SEP]",
                    _pad="[PAD]",
                    candidate_num=_c,
                )
                print(f"candddd result: {_dict}")
                print(_dict)
                f.write(str(_dict))
                f.write("\n")
                f.flush()


if __name__ == "__main__":
    main()
