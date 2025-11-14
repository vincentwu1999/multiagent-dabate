import json, random
from typing import List, Dict, Any, Optional

random.seed(7)

def load_biography_entries(path: str = "biography/article.json", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list): raise ValueError("Biography dataset must be a list of entries.")
        entries = [{"name": x["name"], "article": x.get("article", "")} for x in data if "name" in x]
        random.shuffle(entries)
        return entries[:limit] if limit else entries
    except Exception:
        base = [
            # Foundational & historical
            "Alan Turing", "Alonzo Church", "John von Neumann", "Ada Lovelace", "Grace Hopper",
            "Edsger W. Dijkstra", "C. A. R. Hoare", "John Backus", "Peter Naur", "Niklaus Wirth",
            "Donald Knuth", "Tony Hoare", "Robin Milner", "Dana Scott",

            # Theory & algorithms
            "Stephen Cook", "Richard Karp", "Robert Tarjan", "Leslie Valiant", "Michael O. Rabin",
            "Manuel Blum", "Lenore Blum", "Avi Wigderson", "László Lovász", "Sanjeev Arora",
            "Subhash Khot", "Noga Alon", "Shimon Even", "Moshe Y. Vardi", "Amir Pnueli",
            "Nancy Lynch", "Dana Angluin", "Juris Hartmanis", "John Hopcroft", "David S. Johnson",
            "Éva Tardos", "Jon Kleinberg", "Tim Roughgarden",

            # Cryptography & security
            "Whitfield Diffie", "Martin Hellman", "Ron Rivest", "Adi Shamir", "Leonard Adleman",
            "Shafi Goldwasser", "Silvio Micali", "Cynthia Dwork", "Dan Boneh", "Susan Landau",
            "Ross Anderson",

            # Databases & data systems
            "Michael Stonebraker", "Jim Gray", "David DeWitt", "Hector Garcia-Molina",
            "Jeffrey Ullman", "Alfred Aho", "Andrew Yao",

            # Programming languages & systems
            "Barbara Liskov", "Butler Lampson", "Bjarne Stroustrup", "Dennis Ritchie",
            "Ken Thompson", "Brian Kernighan", "John Ousterhout", "Andrew S. Tanenbaum",
            "Monica Lam", "Guido van Rossum", "Yukihiro Matsumoto", "Alan Kay",

            # AI / ML / Robotics
            "Judea Pearl", "Michael I. Jordan", "Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio",
            "Andrew Ng", "Jürgen Schmidhuber", "Christopher Bishop", "Richard Sutton",
            "Stuart Russell", "Peter Norvig", "Daphne Koller", "Fei-Fei Li", "Jitendra Malik",
            "Sergey Levine", "Leslie Kaelbling", "Rodney Brooks", "Marc Raibert",
            "Christopher Manning", "Dan Jurafsky",

            # HCI / Visualization
            "Ben Shneiderman", "Terry Winograd", "Ivan Sutherland", "Pat Hanrahan", "Ed Catmull",
            "Jim Blinn",

            # Networking / Web
            "Vint Cerf", "Bob Kahn", "Radia Perlman", "Van Jacobson", "Tim Berners-Lee",

            # Graphics / Vision (more)
            "Takeo Kanade", "Pietro Perona", "Bill Freeman", "Deva Ramanan",

            # Bioinformatics / Computational biology
            "David Haussler", "Gene Myers", "Michael Waterman",

            # Software engineering & compilers (additional)
            "Frances Allen", "Barbara Simons", "Margaret Hamilton", "John McCarthy",
        ]
        random.shuffle(base)
        base = base[:limit] if limit else base
        return [{"name": n, "article": ""} for n in base]
