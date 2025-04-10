#!/usr/bin/env python
# coding: utf-8
"""
@Author: lizheng
@Date: 2025-04-02
@Description: https://lizheng.blog.csdn.net/article/details/147091139?spm=1011.2415.3001.5331
具体可以查看博文https://blog.csdn.net/qq_36603091/article/details/147093831?spm=1001.2014.3001.5501
"""
import re 
import collections

print("Libraries 're' and 'collections' imported.")



corpus_raw = """
Alice was beginning to get very tired of sitting by her sister on the
bank, and of having nothing to do: once or twice she had peeped into the
book her sister was reading, but it had no pictures or conversations in
it, 'and what is the use of a book,' thought Alice 'without pictures or
conversation?'
So she was considering in her own mind (as well as she could, for the
hot day made her feel very sleepy and stupid), whether the pleasure
of making a daisy-chain would be worth the trouble of getting up and
picking the daisies, when suddenly a White Rabbit with pink eyes ran
close by her.
"""

print(f"Training corpus defined (length: {len(corpus_raw)} characters).")





# Convert the raw corpus to lowercase
corpus_lower = corpus_raw.lower()

# Optional: Display a snippet of the lowercased corpus
# print("Lowercased corpus snippet:")
# print(corpus_lower[:100]) 
print("Corpus converted to lowercase.")






# Define the regular expression for splitting words and punctuation
split_pattern = r'\w+|[^\s\w]+'

# Apply the regex to the lowercased corpus to get a list of initial tokens
initial_word_list = re.findall(split_pattern, corpus_lower)

print(f"Corpus split into {len(initial_word_list)} initial words/tokens.")
# Optional: Display the first few tokens
print(f"First 3 initial tokens: {initial_word_list[:3]}")





# Use collections.Counter to count frequencies of items in initial_word_list
word_frequencies = collections.Counter(initial_word_list)

print(f"Calculated frequencies for {len(word_frequencies)} unique words/tokens.")
# Display the 3 most frequent tokens and their counts
print("3 Most frequent tokens:")
for token, count in word_frequencies.most_common(3):
    print(f"  '{token}': {count}")




# Define the special end-of-word symbol
end_of_word_symbol = '</w>'

# Create a dictionary to hold the initial representation of the corpus
# Key: original unique word/token, Value: list of characters + end_of_word_symbol
initial_corpus_representation = {}

# Iterate through the unique words/tokens identified by the frequency counter
for word in word_frequencies:
    # Convert the word string into a list of its characters
    char_list = list(word)
    # Append the end-of-word symbol to the list
    char_list.append(end_of_word_symbol)
    # Store this list in the dictionary with the original word as the key
    initial_corpus_representation[word] = char_list

print(f"Created initial corpus representation for {len(initial_corpus_representation)} unique words/tokens.")
# Optional: Display the representation for a sample word
example_word = 'beginning'
if example_word in initial_corpus_representation:
    print(f"Representation for '{example_word}': {initial_corpus_representation[example_word]}")
example_punct = '.'
if example_punct in initial_corpus_representation:
    print(f"Representation for '{example_punct}': {initial_corpus_representation[example_punct]}")




# Initialize an empty set to store the unique initial symbols (vocabulary)
initial_vocabulary = set()

# Iterate through the character lists stored in the initial corpus representation
for word in initial_corpus_representation:
    # Get the list of symbols for the current word
    symbols_list = initial_corpus_representation[word]
    # Update the vocabulary set with the symbols from this list
    # The `update` method adds all elements from an iterable (like a list) to the set
    initial_vocabulary.update(symbols_list)

# Although update should have added '</w>', we can explicitly add it for certainty
# initial_vocabulary.add(end_of_word_symbol)

print(f"Initial vocabulary created with {len(initial_vocabulary)} unique symbols.")
# Optional: Display the sorted list of initial vocabulary symbols
print(f"Initial vocabulary symbols: {sorted(list(initial_vocabulary))}")


num_merges = 75 # Let's use 75 merges for this example

# Initialize an empty dictionary to store the learned merge rules
# Format: { (symbol1, symbol2): merge_priority_index }
learned_merges = {}

# Create a working copy of the corpus representation to modify during training
# Using .copy() ensures we don't alter the original initial_corpus_representation
current_corpus_split = initial_corpus_representation.copy()

# Create a working copy of the vocabulary to modify during training
current_vocab = initial_vocabulary.copy()

print(f"Training state initialized. Target number of merges: {num_merges}")
print(f"Initial vocabulary size: {len(current_vocab)}")





# Start the main loop that iterates for the specified number of merges
print(f"\n--- Starting BPE Training Loop ({num_merges} iterations) ---")
for i in range(num_merges):
    # --- Code for steps 2.3 to 2.9 will go inside this loop --- 
    # Print the current iteration number (starting from 1)
    print(f"\nIteration {i + 1}/{num_merges}")


    print("  Step 2.3: Calculating pair statistics...")
    pair_counts = collections.Counter()
    # Iterate through the original words and their frequencies
    for word, freq in word_frequencies.items():
        # Get the *current* list of symbols for this word from the evolving representation
        symbols = current_corpus_split[word]
        # Iterate through the adjacent pairs in the current symbol list
        # Loop from the first symbol up to the second-to-last symbol
        for j in range(len(symbols) - 1):
            # Form the pair (tuple of two adjacent symbols)
            pair = (symbols[j], symbols[j+1])
            # Increment the count for this pair by the frequency of the original word
            pair_counts[pair] += freq 
    print(f"  Calculated frequencies for {len(pair_counts)} unique pairs.")

    print("  Step 2.4: Checking if pairs exist...")
    if not pair_counts:
        print("  No more pairs found to merge. Stopping training loop early.")
        break # Exit the 'for i in range(num_merges)' loop
    print("  Pairs found, continuing training.")

.
    print("  Step 2.5: Finding the most frequent pair...")
    try:
        best_pair = max(pair_counts, key=pair_counts.get)
        best_pair_frequency = pair_counts[best_pair]
        print(f"  Found best pair: {best_pair} with frequency {best_pair_frequency}")
    except ValueError:
        # This should theoretically be caught by the 'if not pair_counts' check above,
        # but adding robust error handling is good practice.
        print("  Error: Could not find maximum in empty pair_counts. Stopping.")
        break

 
    # correctly tokenizing new text later.
    print(f"  Step 2.6: Storing merge rule (Priority: {i})...")
    learned_merges[best_pair] = i
    print(f"  Stored: {best_pair} -> Priority {i}")

    # --- Step 2.7 (Inside Loop): Create New Symbol ---
    # Theory: The new symbol representing the merged pair is created by simply 
    # concatenating the string representations of the two symbols in the pair.
    print("  Step 2.7: Creating new symbol from best pair...")
    new_symbol = "".join(best_pair)
    print(f"  New symbol created: '{new_symbol}'")


    print("  Step 2.8: Updating corpus representation...")
    next_corpus_split = {}
    # Iterate through all original words (keys in the current split dictionary)
    for word in current_corpus_split:
        # Get the list of symbols for this word *before* applying the current merge
        old_symbols = current_corpus_split[word]
        # Initialize an empty list to build the new sequence of symbols for this word
        new_symbols = []
        # Initialize scan index for the old_symbols list
        k = 0
        # Scan through the old symbols list
        while k < len(old_symbols):
            # Check if we are not at the very last symbol (to allow pair formation)
            # and if the pair starting at index k matches the best_pair to be merged
            if k < len(old_symbols) - 1 and (old_symbols[k], old_symbols[k+1]) == best_pair:
                # If match found, append the new merged symbol to our new list
                new_symbols.append(new_symbol)
                # Advance the scan index by 2 (skipping both parts of the merged pair)
                k += 2
            else:
                # If no match, just append the current symbol from the old list
                new_symbols.append(old_symbols[k])
                # Advance the scan index by 1
                k += 1
        # Store the newly constructed symbol list for this word in the temporary dictionary
        next_corpus_split[word] = new_symbols
        
    # After processing all words, update the main corpus split to reflect the merge
    current_corpus_split = next_corpus_split
    print("  Corpus representation updated for all words.")

    print("  Step 2.9: Updating vocabulary...")
    current_vocab.add(new_symbol)
    print(f"  Added '{new_symbol}' to vocabulary. Current size: {len(current_vocab)}")


print(f"\n--- BPE Training Loop Finished after {i + 1} iterations (or target reached) ---")

# Assign final state variables for clarity (optional, could just use current_*)
final_vocabulary = current_vocab
final_learned_merges = learned_merges
final_corpus_representation = current_corpus_split

print("Final vocabulary, merge rules, and corpus representation are ready.")





print("--- Inspecting Training Results ---")
print(f"Final Vocabulary Size: {len(final_vocabulary)} tokens")





print("\nLearned Merge Rules (Sorted by Priority):")

# Convert the dictionary items to a list of (pair, priority) tuples
merges_list = list(final_learned_merges.items())

# Sort the list based on the priority (the second element of the tuple, index 1)
# `lambda item: item[1]` tells sort to use the priority value for sorting
sorted_merges_list = sorted(merges_list, key=lambda item: item[1])

# Display the sorted merges
print(f"Total merges learned: {len(sorted_merges_list)}")
# Print a sample if the list is long, otherwise print all
display_limit = 20 
if len(sorted_merges_list) <= display_limit * 2:
    for pair, priority in sorted_merges_list:
        print(f"  Priority {priority}: {pair} -> '{''.join(pair)}'")
else:
    print("  (Showing first 10 and last 10 merges)")
    # Print first N
    for pair, priority in sorted_merges_list[:display_limit // 2]:
        print(f"  Priority {priority}: {pair} -> '{''.join(pair)}'")
    print("  ...")
    # Print last N
    for pair, priority in sorted_merges_list[-display_limit // 2:]:
        print(f"  Priority {priority}: {pair} -> '{''.join(pair)}'")




print("\nFinal Representation of Example Words from Training Corpus:")

# List some words we expect to see interesting tokenization for
example_words_to_inspect = ['beginning', 'conversations', 'sister', 'pictures', 'reading', 'alice']

for word in example_words_to_inspect:
    if word in final_corpus_representation:
        print(f"  '{word}': {final_corpus_representation[word]}")
    else:
        print(f"  '{word}': Not found in original corpus (should not happen if chosen from corpus).")


new_text_to_tokenize = "Alice thought reading was tiresome without pictures."

print(f"--- Tokenizing New Text ---")
print(f"Input Text: '{new_text_to_tokenize}'")





print("Step 4.2: Preprocessing the new text...")

new_text_lower = new_text_to_tokenize.lower()
print(f"  Lowercased: '{new_text_lower}'")


new_words_list = re.findall(split_pattern, new_text_lower)
print(f"  Split into words/tokens: {new_words_list}")



# Initialize an empty list to store the final sequence of tokens for the whole text
tokenized_output = []
print("Step 4.3: Initialized empty list for tokenized output.")




print("Step 4.4: Starting iteration through words of the new text...")
# Loop through each preprocessed word/token from the new text
for word in new_words_list:
    print(f"\n  Processing Word: '{word}'")
    

    print("    Step 4.5: Initializing symbols for this word...")
    word_symbols = list(word) + [end_of_word_symbol] # Recall: end_of_word_symbol = '</w>'
    print(f"      Initial symbols: {word_symbols}")
    

    print("    Step 4.6: Applying learned merges iteratively...")
    while True: # Loop until no more merges can be applied to this word
        best_priority_found_this_pass = float('inf') 
        pair_to_merge_this_pass = None
        merge_location_this_pass = -1
        

        scan_index = 0
        while scan_index < len(word_symbols) - 1:
            # Form the adjacent pair at the current scan index
            current_pair = (word_symbols[scan_index], word_symbols[scan_index + 1])
            
            # Check if this pair exists in our learned merge rules
            if current_pair in final_learned_merges:
                # If it exists, get its priority (when it was learned)
                current_pair_priority = final_learned_merges[current_pair]
                
                # Check if this pair's priority is better (lower number) than the best found so far *in this pass*
                if current_pair_priority < best_priority_found_this_pass:
                    # If yes, update the best merge found for this pass
                    best_priority_found_this_pass = current_pair_priority
                    pair_to_merge_this_pass = current_pair
                    merge_location_this_pass = scan_index # Record where the best merge starts
                    
            # Move to the next position to check the next pair
            scan_index += 1
            

        if pair_to_merge_this_pass is not None:
            # An applicable merge was found. Apply the one with the highest priority.
            merged_symbol = "".join(pair_to_merge_this_pass)
            print(f"      Applying highest priority merge: {pair_to_merge_this_pass} (Priority {best_priority_found_this_pass}) at index {merge_location_this_pass} -> '{merged_symbol}'")
            
            # Reconstruct the word_symbols list with the merge applied
            # Slice before the merge + new symbol + slice after the merge
            word_symbols = word_symbols[:merge_location_this_pass] + [merged_symbol] + word_symbols[merge_location_this_pass + 2:]
            print(f"      Updated symbols: {word_symbols}")
            # Continue to the next iteration of the 'while True' loop to scan the *updated* symbols list
        else:
            # No applicable merges were found in the entire scan of the current symbols list
            print("      No more applicable learned merges found for this word.")
            break # Exit the 'while True' loop for this word
            

    print("    Step 4.7: Appending final tokens for this word to overall output...")
    tokenized_output.extend(word_symbols)
    print(f"      Appended: {word_symbols}")

print("\nFinished processing all words in the new text.")




print("\n--- Final Tokenization Result ---")
print(f"Original Input Text: '{new_text_to_tokenize}'")
print(f"Tokenized Output ({len(tokenized_output)} tokens): {tokenized_output}")

