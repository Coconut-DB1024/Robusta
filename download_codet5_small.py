import hashlib
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen


MODEL_ID = "Salesforce/codet5-small"
REVISION = "b1ee9570c289f21b5922b9c768a1ce12957bf968"
MODEL_DIR = Path(__file__).resolve().parent / "src" / "codet5-small"
INSTALL_COMMAND = (
    'python -m pip install "torch>=2.6" "transformers==4.57.1" '
    '"tokenizers==0.22.1" "safetensors==0.6.2"'
)
SOURCE_HASHES = {
    "config.json": "c067a6377fc235b103a1577fcb8a114e2885b700198bad22e34f84659d0025ca",
    "merges.txt": "5d346f84939a98df0cde902df7d60154b461cde1b90dc653ab9b81d74e752a4d",
    "vocab.json": "43bb485f4de0f2fd49b370bef4efab23ea3ab6d0019e72bdd9cec1436a6eaa2b",
    "special_tokens_map.json": "2454193b9f88437a3c0497300f21a2a4372115b5367ea71f07e912589baedaa6",
    "tokenizer_config.json": "ccc2672ac229eef1a36010a147b8d3e7881d39a30b9bca7ddaa73a30fccfd517",
    "added_tokens.json": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    "pytorch_model.bin": "968fb0f45e1efc8cf3dd50012d1f82ad82098107cbadde2c0fdd8e61bac02908",
}
OUTPUT_HASHES = {
    "config.json": "e4b9e0e6c7fda466e82616791b20d7d6a43f3a8e41b1210b2bee4734b59cdbdb",
    "generation_config.json": "b07d18c156603831ffd32e297d1d93d2af7d95804b96ec7bdff7d780649cacd9",
    "merges.txt": "5d346f84939a98df0cde902df7d60154b461cde1b90dc653ab9b81d74e752a4d",
    "model.safetensors": "967f7e6d35b83befa768c99ef7179eaa967db98a72e6193f43b2712cb51ab75c",
    "special_tokens_map.json": "2398bfc908dd21969fd2f589130c4dfb1b8c3a15c2644cd16b1cc38ecbce3d2b",
    "tokenizer.json": "d5e8316ab01a7eba95a445df95c58cd34b493bee83b9f213accc9e03a8f573c2",
    "tokenizer_config.json": "cf849eef873340d3ce6b6cd5ebce2536bdb1f2c23c407d6165d3b24d8a186442",
    "vocab.json": "8f1f0353a67cc54521111c8ca10f214ca4ccb8018b79126448e0ea95078c692e",
}
CRLF_FILES = {
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(path):
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return data.replace(b"\n", b"\r\n") if path.name in CRLF_FILES else data


def output_sha256(path):
    if path.name == "model.safetensors":
        return file_sha256(path)
    return hashlib.sha256(normalized_text(path)).hexdigest()


def missing_files(destination):
    missing = []
    for name, expected in OUTPUT_HASHES.items():
        path = destination / name
        if not path.exists():
            missing.append(name)
        elif not path.is_file() or output_sha256(path) != expected:
            raise RuntimeError(
                f"Existing file differs from the verified model: {path}\n"
                "Move it aside before downloading; it will not be overwritten."
            )
    return missing


def load_dependencies():
    try:
        from packaging.version import Version

        for package, required in (
            ("transformers", "4.57.1"),
            ("tokenizers", "0.22.1"),
            ("safetensors", "0.6.2"),
        ):
            if version(package) != required:
                raise RuntimeError(f"{package}=={required} is required")
        if Version(version("torch")) < Version("2.6"):
            raise RuntimeError("torch>=2.6 is required")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(f"{exc}\nInstall the download dependencies:\n{INSTALL_COMMAND}") from exc
    return AutoTokenizer, AutoModelForSeq2SeqLM


def download_file(name, destination):
    print(f"Downloading {name} ...", flush=True)
    request = Request(
        f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}",
        headers={"User-Agent": "Robusta-model-download"},
    )
    digest = hashlib.sha256()
    with urlopen(request, timeout=60) as response, destination.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != SOURCE_HASHES[name]:
        raise RuntimeError(f"Downloaded file failed SHA-256 verification: {name}")


def download_model(destination=MODEL_DIR):
    destination = Path(destination).resolve()
    missing = missing_files(destination)
    if not missing:
        print(f"All 8 model files are already present and verified: {destination}")
        return

    tokenizer_class, model_class = load_dependencies()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".codet5-download-", dir=destination.parent) as temporary:
        source = Path(temporary) / "source"
        output = Path(temporary) / "output"
        source.mkdir()
        for name in SOURCE_HASHES:
            download_file(name, source / name)

        print("Saving and verifying the 8 model files ...", flush=True)
        tokenizer = tokenizer_class.from_pretrained(source, local_files_only=True)
        model = model_class.from_pretrained(
            source, local_files_only=True, use_safetensors=False
        )
        tokenizer.save_pretrained(output)
        model.save_pretrained(output, safe_serialization=True)
        if {path.name for path in output.iterdir()} != set(OUTPUT_HASHES):
            raise RuntimeError("The generated model file list does not match the expected 8 files.")
        for name, expected in OUTPUT_HASHES.items():
            if name != "model.safetensors":
                path = output / name
                path.write_bytes(normalized_text(path))
            if file_sha256(output / name) != expected:
                raise RuntimeError(f"Generated file failed SHA-256 verification: {name}")

        missing = missing_files(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for name in missing:
            (output / name).replace(destination / name)

    print(f"All 8 model files have been verified and saved to: {destination}")


if __name__ == "__main__":
    try:
        download_model()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        sys.exit(1)
