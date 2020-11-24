'''
Michael Lepori
9/11/19

CFG Sentence Generator
'''

import argparse
from numpy.random import choice
import numpy as np

def generate_sentence(grammar, sentence, call_count, tree):

    nt_exists = False

    # Check if the recursion limit is reached
    recursion_limit = False
    if call_count >= 450:
        recursion_limit = True

    # Check for nonterminals
    for word in sentence:
        if word in grammar.keys():
            nt_exists = True
    
    # Base case: no nonterminals
    if nt_exists == False:
        return sentence
    
    else:
        # Move horizontally
        found_nt = False
        for i in range(len(sentence)):
            if found_nt == True:
                continue

            # Find first nonterminal
            if sentence[i] in grammar.keys():
                found_nt = True

                left = sentence[:i]
                right = sentence[i+1:]

                # Choose rule and recurse further
                if recursion_limit:
                    rule_choice = ['...']
                else:
                    nt = sentence[i]
                    rule_idx = choice(len(grammar[nt]), p=probs[nt])
                    rule_choice = grammar[nt][rule_idx]
                    
                    if tree:    # Includes derivation if tree function is on
                        nt_token = "(" + nt     # Paren removes nonterminal from list of nonterminals
                        rule_choice = [nt_token] + rule_choice + [")"]  # Appends close paren with a 
                                                                        # space such that nonterminals 
                                                                        # introduced by the rule can be processed. 
                                                                        # This formatting is fixed right before printing


                new_sent = left + rule_choice
                new_sent = new_sent + right

                return generate_sentence(grammar, new_sent, call_count + 1, tree)


if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("grammar_file")
    parser.add_argument("numsent", type=int, nargs="?", default=1)
    parser.add_argument("-t", "--tree", action="store_true")
    args = parser.parse_args()

    # Parse file and grammar
    grammar = {}
    probs = {}

    f = open(args.grammar_file, 'r')
    for line in f:
        if "#" in line:
            idx = line.index("#")
            line = line[:idx]
        tokens = line.split()

        if len(tokens) == 0:
            continue

        odds = float(tokens[0])
        lhs = tokens[1]
        rhs = tokens[2:]

        if lhs not in grammar.keys():
            grammar[lhs] = []
            probs[lhs] = []

        grammar[lhs].append(rhs)
        probs[lhs].append(odds)

    # Load in correct probabilities into 
    for key in probs.keys():
        key_probs = probs[key]
        total = sum(key_probs)
        key_probs = [prob / total for prob in key_probs]
        probs[key] = np.array(key_probs)

    # Generate sentences
    for i in range(args.numsent):
        sentence = generate_sentence(grammar, ["ROOT"], 0, args.tree)
        sentence = ' '.join(sentence)
        sentence = sentence.replace(" )", ")") # Formats tree string for pretty print
        print(sentence)
    

    



