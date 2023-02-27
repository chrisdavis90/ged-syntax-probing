import argparse
from tqdm import tqdm

# Script modified from Chris Bryant
# Convert an .m2 file to .conll

def main(m2_file, out, coder=0, bin=False, op=False, main=False, all=False):
    if not(bin or op or main or all):
        print('must set one of: bin, op, main, all')
        exit()

    # Parse args
    # args = parse_args()
    # Open output file
    out_conll = open(out, "w")
    # Test to make sure the input coder exists
    found = False

    # Load the whole M2 file
    m2 = open(m2_file).read().strip().split("\n\n")
    # Loop through M2 blocks
    for m2_block in tqdm(m2):
        # Preprocessing
        m2_block = m2_block.split("\n")
        orig = m2_block[0].split()[1:] # Ignore "S"
        edits = m2_block[1:]
        # Store detection ids here
        tok_dict = {}
        # Loop through edits
        for e in edits:
            e = e.split("|||")
            # Ignore edits from different coders
            if int(e[-1]) != coder: continue
            else: found = True
            # Ignore noops
            if e[1] == "noop": continue
            # Ignore UNK in the non-binary setting; we don't know the type
            if e[1] == "UNK" and not bin: continue
            # Get edit info
            start = int(e[0].split()[1])
            end = int(e[0].split()[2])
            cat = e[1] # Only get the MRU part of the type
            # Missing words
            if start == end:
                if start not in tok_dict: tok_dict[start] = [cat]
                else: tok_dict[start].append(cat)
            # Replacement and Unnecessary words
            else:
                # Get all the tokens in the edit
                ids = list(range(start, end))
                # Loop through the ids
                for i in ids:
                    if i not in tok_dict: tok_dict[i] = [cat]
                    else: tok_dict[i].append(cat)
        # Loop through orig tokens
        for i in range(0, len(orig)):
            # Edited tokens
            if i in tok_dict:
                # Binary labels
                if bin:
                    out_conll.write("\t".join([orig[i], "I"]))
                # Operation error type
                elif op:
                    # Collapse overlapping labels to a single label
                    ops = list(set([op[0] for op in tok_dict[i]]))
                    ops = ops[0] if len(ops) == 1 else "R"
                    out_conll.write("\t".join([orig[i], ops]))
                # Main error type
                elif main:
                    # For overlapping labels, choose the last
                    out_conll.write("\t".join([orig[i], tok_dict[i][-1][2:]]))
                # Full error type
                elif all:
                    # For overlapping labels, choose the last
                    out_conll.write("\t".join([orig[i], tok_dict[i][-1]]))                
            # Correct tokens
            else: 
                out_conll.write("\t".join([orig[i], "C"]))
            # Newline after each token
            out_conll.write("\n")
        # Write new line at the end of the block
        out_conll.write("\n")

    # Warning if there are no edits for the input coder
    if not found:
        print("\nWARNING: No edits found for coder "+str(coder)+".\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert M2 to CoNLL format.")
    parser.add_argument("m2_file", help="Path to a M2 file.")
    parser.add_argument("-out", help="The output filepath.", required=True)
    parser.add_argument("-coder", help="Choose a single annotator.", default=0, type=int)
    type_group = parser.add_mutually_exclusive_group(required=True)
    type_group.add_argument("-bin", help = "Binary detection labels (I, C)", action = "store_true")
    type_group.add_argument("-op", help = "Operation detection labels (M, R, U, C)", action = "store_true")
    type_group.add_argument("-main", help = "Main detection labels (e.g. DET)", action = "store_true")
    type_group.add_argument("-all", help = "All detection labels (e.g. M:DET)", action = "store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    main(
        m2_file=args.m2_file,
        out=args.out,
        bin=args.bin,
        op=args.op,
        main=args.main,
        all=args.all
    )
