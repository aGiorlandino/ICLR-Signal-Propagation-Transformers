from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

dataset = load_dataset("roneneldan/TinyStories", split="train")

def get_text_iterator():
    for item in dataset:
        yield item["text"]

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = WordPieceTrainer(
    vocab_size=2000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

print("Training tokenizer...")
tokenizer.train_from_iterator(get_text_iterator(), trainer=trainer)


tokenizer.save("tiny-tokenizer.json")
print("Tokenizer saved!")