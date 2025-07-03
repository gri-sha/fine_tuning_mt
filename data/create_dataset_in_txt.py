MIN_LENGTH = 10
TARGET_PATH = "data/all.txt"

if __name__ == "__main__":
    target = open(TARGET_PATH, "w")
    total_written = 0
    total_read = 0

    for i in range(1, 5):
        en = open(f"data/en/book{i}_en.txt", "r")
        fr = open(f"data/fr/book{i}_fr.txt", "r")

        book_lines_read = 0

        while True:
            en_line = en.readline()
            fr_line = fr.readline()

            if not en_line or not fr_line:
                break
            
            book_lines_read += 1

            en_line = en_line.strip()
            fr_line = fr_line.strip()

            if len(en_line) < MIN_LENGTH or len(fr_line) < MIN_LENGTH:
                continue

            target.write(en_line + " || " + fr_line + "\n")
            total_written += 1

        total_read += book_lines_read
        print(f"Book {i}: {book_lines_read} lines written.")

    print()
    print(f"Total lines read: {total_read}")
    print(f"Total lines written: {total_written}")

