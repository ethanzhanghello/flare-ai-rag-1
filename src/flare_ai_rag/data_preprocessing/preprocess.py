import os
import json
import re
import csv


def clean_text(text: str) -> str:
    """
    Removes unnecessary symbols, extra spaces, and formats text properly.
    :param text: The text to be cleaned.
    :return: Cleaned text as a string.
    """
    text = re.sub(r'\s+', ' ', text)  # Removes extra spaces
    text = re.sub(r'[^a-zA-Z0-9.,?!:;()"\s]', '', text)  # Removes weird symbols
    return text.strip()


def split_text(text: str, chunk_size: int = 500) -> list[str]:
    """
    Splits a document into smaller chunks, ensuring sentences are not cut.
    :param text: The text to split.
    :param chunk_size: The maximum size of each chunk.
    :return: A list of smaller text chunks.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk)) + len(word) < chunk_size:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def preprocess_documents(input_folder: str = "data", output_folder: str = "processed_data") -> None:
    """
    Reads CSV files, extracts text, cleans, splits into chunks, and stores metadata.
    :param input_folder: Folder where the CSV files are stored.
    :param output_folder: Folder where processed chunks will be stored.
    """
    os.makedirs(output_folder, exist_ok=True)
    metadata = []

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        # Process CSV files
        if filename.endswith(".csv"):
            with open(filepath, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if "content" in row:  # Ensure content column exists
                        text = clean_text(row["content"])  # Extract and clean text
                        title = row.get("title", "Unknown Title")
                        author = row.get("author", "Unknown Author")
                        date = row.get("date", "Unknown Date")

                        chunks = split_text(text)
                        for idx, chunk in enumerate(chunks):
                            chunk_filename = f"{filename}_chunk{idx}.txt"
                            chunk_path = os.path.join(output_folder, chunk_filename)

                            with open(chunk_path, "w", encoding="utf-8") as chunk_file:
                                chunk_file.write(chunk)

                            metadata.append({
                                "filename": chunk_filename,
                                "original": filename,
                                "title": title,
                                "author": author,
                                "date": date
                            })

    # Save metadata
    with open(os.path.join(output_folder, "metadata.json"), "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=4)

    print("Preprocessing complete!")


if __name__ == "__main__":
    preprocess_documents()
