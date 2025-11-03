from pathlib import Path
import argparse
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--corpus_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--min_frequency", type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus_files = sorted(args.corpus_dir.glob("*.train"))
    if not corpus_files:
        raise FileNotFoundError(f"No .train files found under {args.corpus_dir}")

    for f in corpus_files:
        print(f"  - {f.name}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(f) for f in corpus_files],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )

    tokenizer.save_model(str(args.output_dir))
    print(f"Raw tokenizer files saved to {args.output_dir}")

    fast_tokenizer = RobertaTokenizerFast.from_pretrained(
        str(args.output_dir),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    fast_tokenizer.save_pretrained(args.output_dir)
    print(f"HuggingFace tokenizer saved to: {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
