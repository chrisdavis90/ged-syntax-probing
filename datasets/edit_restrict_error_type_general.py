import argparse
from tqdm import tqdm

# Script modified from Chris Bryant
# Generate a new M2 file that includes a subset of error-operations (e.g only include replacement-errors).
# Example: Given a sentence with 3 edits made up of 1 replacement and 2 missing errors,
# We can correct the 2 missing errors and leave the replacement error

def main(m2_file: str, out_file: str, restricted_error: str, verbose: bool=False):

    # Load whole M2 file and split it into blocks
    m2 = open(m2_file).read().strip().split("\n\n")
    # Ignore edits with these error types
    ignored = {"noop", "UNK", "Um"}
    # Output combinations M2 file
    out = open(out_file, "w")

    restricted_error_type = restricted_error

    removed_edits = 0
    total_edits = 0
    removed_sentences = 0
    total_sentences = 0

    # Loop through M2 sentence+edit blocks
    for block in tqdm(m2):
        block = block.split("\n")
        
        # Get the original sentence as a list of tokens
        o_sent = block[0].split()[1:] # Ignores "S"
        # Get simplified edits for the target annotator
        edits = simplify_edits(block[1:], ignored, 0)

        total_sentences += 1
        total_edits += len(edits)

        # pre-process to remove overlapping edits?
        tok_dict = {}
        for edit_i, e in enumerate(edits):
            # Missing words
            start = e[0]
            end = e[1]
            cat = e[2]

            if start == end:
                if start not in tok_dict: tok_dict[start] = [edit_i]
                else: tok_dict[start].append(edit_i)
            # Replacement and Unnecessary words
            else:
                # Get all the tokens in the edit
                ids = list(range(start, end))
                # Loop through the ids
                for i in ids:
                    if i not in tok_dict: tok_dict[i] = [edit_i]
                    else: tok_dict[i].append(edit_i)

        # find edits to remove
        edits_to_remove = []
        for token_i, edits_indices in tok_dict.items():
            # if there are multiple edits for this token,
            #   remove all except for the {restricted_error_type} edits
            # if there are multiple {restricted_error_type} edits for this token,
            #   keep the last
            if len(edits_indices) > 1:
                error_indices = [edit_index for edit_index in edits_indices if edits[edit_index][2] == restricted_error_type]
                if len(error_indices) > 1:
                    error_indices = error_indices[-1:]  # keep the last

                remove_indices = set(edits_indices)
                remove_indices = remove_indices.difference(set(error_indices))
                
                edits_to_remove.extend(list(remove_indices))
            
            else:
                # remove edit if it isn't a {restricted_error_type} error
                error_type = edits[edits_indices[0]][2]

                if error_type != restricted_error_type:
                    edits_to_remove.append(edits_indices[0])

        # # Represent edits as a set of token ids
        all_edit_ids = set(range(0, len(edits)))
        keep_edit_ids = all_edit_ids.difference(edits_to_remove)
        
        # Create a copy of orig to be corrected based on not_c
        c_sent = o_sent[:]
        
        # Track offsets per edit
        offsets = [0] * len(edits)
        
        # Loop through the edits ids NOT in the combination
        for e in sorted(list(set(edits_to_remove))):
        # for e in edits_to_remove:
            # Offsets for an edit include only those edits
            #   which took place *before* this one
            start_offset = edits[e][0] + sum(offsets[:e])
            end_offset = edits[e][1] + sum(offsets[:e])

            # Apply the edit: [start, end, cat, cor]
            c_sent[start_offset : end_offset] = edits[e][3]
            
            # Update the offset
            offsets[e] = (edits[e][0] - edits[e][1]) + len(edits[e][3])

        removed_edits += len(edits_to_remove)
        if len(keep_edit_ids) == 0:
            removed_sentences += 1

        # If there are remaining edits in the sentence,
        #   write the partially corrected sentence to the output file
        if len(keep_edit_ids) > 0:
            out.write(" ".join(["S"]+c_sent)+"\n")
            # Loop through the edit ids we keep
            for e in keep_edit_ids:
                # Create a copy and modify start/end positions
                #   so that we don't permanently change the positions, 
                #   in case we have more combinations to go through
                target_edit = list(edits[e])
                target_edit[0] += sum(offsets[:e])
                target_edit[1] += sum(offsets[:e])

                # Format the edit as an M2 string
                e = format_edit(target_edit)
                # Write it to the output file
                out.write(e+"\n")
            # Write a new line at the end of all edits in this combination
            out.write("\n")

    if verbose:
        print(f'Total sentences: {total_sentences}')
        print(f'Removed sentences: {removed_sentences}')
        
        print(f'Total edits: {total_edits}')  
        print(f'Removed edits: {removed_edits}')

    return total_sentences, removed_sentences, total_edits, removed_edits

def create_args():
    # Define and parse program input
    parser = argparse.ArgumentParser()
    parser.add_argument("m2_file", help="The path to a M2 file.")
    parser.add_argument("-o", "--out", help="The name of the output M2 file.", required=True)
    parser.add_argument("-e", help="The error type to restrict to. Must be an operation + main type. For example, R:VERB:SVA.", type=str, required=True)
    parser.add_argument("-v", help="Verbose: whether to print informative statements. Default: False", type=bool, default=False)
    return parser


def parse_args():
    # Parse command line arguments
    parser = create_args()
    args = parser.parse_args()
    return args


def simplify_edits(edits, ignored, id):
    # Input 1: A list of edit strings from an M2 file
    # Input 2: A list set of error types to be ignored
    # Input 3: The id of the target annotator; edits by other annotators are ignored.
    # Output: A list of edit sublists for the target annotator: [start, end, cat, cor]
    new_edits = []
    for e in edits:
        e = e.split("|||")
        if e[1] in ignored:  continue  #  Ignore noop, UNK, Um types
        annotator = int(e[-1])
        if annotator != id:  continue  #  Ignore non-target annotators
        span = e[0].split()[1:]  #  Ignore "A "
        start = int(span[0])
        end = int(span[1])
        # Save new edit
        new_edits.append([start, end, e[1], e[2].split()])
    return new_edits


def format_edit(edit):
    # Input: A simplified edit: [start, end, cat, cor]
    # Output: The same edit as an M2 string
    span = " ".join(["A", str(edit[0]), str(edit[1])])
    return "|||".join([span, edit[2], " ".join(edit[3]), "REQUIRED", "-NONE-", "0"])


if __name__ == "__main__":
    # Command line args
    args = parse_args()

    main(
        m2_file=args.m2_file,
        out_file = args.out,
        restricted_error= args.e
    )