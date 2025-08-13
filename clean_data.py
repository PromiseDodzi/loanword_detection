import csv

def clean_data(input_name, output_name):
    with open(input_name, "r", encoding="utf-8") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        fieldnames = reader.fieldnames

        with open(output_name, "w", encoding="utf-8", newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()

            for row in reader:

                row["ipa"] = row["ipa"].replace("ː", "").replace("ˈ", "").replace(" ", "").lower()

                for key in row:
                    if key != "ipa" and key != "label":
                        row[key] = row[key].lower()

                writer.writerow(row)

if __name__== "__main__":
    input_name="datasets/data.tsv"
    output_name="datasets/cleaned_data.tsv"
    clean_data(input_name, output_name)