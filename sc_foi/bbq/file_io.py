import csv


def get_delimiter_from_file(filename: str) -> str:
    with open(filename, "r") as fid:
        str_data = fid.read()
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(str_data, delimiters=";,t\t")
            delimiter = dialect.delimiter
            return delimiter
        except csv.Error as err:
            print(f"Error occurred while reading delimiter:\n{err}")
            return ""
