import os
import difflib
import pandas as pd
import ast
import re
import glob2
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()
    return content

def ast_sim(file1_content, file2_content):
    file1_ast = ast.parse(file1_content)
    file2_ast = ast.parse(file2_content)

    file1_nodes = [node.__class__.__name__ for node in ast.walk(file1_ast)]
    file2_nodes = [node.__class__.__name__ for node in ast.walk(file2_ast)]

    file1nodes = " ".join(file1_nodes)
    file2nodes = " ".join(file2_nodes)

    similarity_score = 1 - levenshtein_distance(file1nodes, file2nodes)

    return similarity_score

def extract_functions(code):
    functions = []
    code_lines = code.split("\n")
    for line in code_lines:
        if line.strip().startswith("def"):
            functions.append(line.strip())
    return functions

def compare_function_names(func1, func2):
    func11 = " ".join(func1)
    func21 = " ".join(func2)

    similarity_score = 1 - levenshtein_distance(func11, func21)

    return similarity_score

def compare_function_arguments(func1, func2):
    arguments1 = func1[func1.index("(")+1:func1.index(")")]
    arguments2 = func2[func2.index("(")+1:func2.index(")")]

    arg1 = " ".join(arguments1)
    arg2 = " ".join(arguments2)

    similarity_score = 1 - levenshtein_distance(arg1, arg2)

    return similarity_score

def compare_function_calls(func1, func2):
    call1 = func1[func1.index("("):]
    call2 = func2[func2.index("("):]

    call_1 = " ".join(call1)
    call_2 = " ".join(call2)

    similarity_score = 1 - levenshtein_distance(call_1, call_2)

    return similarity_score

def function_sim(file1_content, file2_content):
    file1_functions = extract_functions(file1_content)
    file2_functions = extract_functions(file2_content)

    if file1_functions != [] and file2_functions != []:
        num_functions_similarity = compare_function_names(str(len(file1_functions)), str(len(file2_functions)))

        function_similarity_scores = []
        for function1 in file1_functions:
            for function2 in file2_functions:

                function_name_similarity = compare_function_names(function1, function2)
                function_arguments_similarity = compare_function_arguments(function1, function2)
                function_calls_similarity = compare_function_calls(function1, function2)

                overall_similarity_score = (function_name_similarity + function_arguments_similarity + function_calls_similarity) / 3
                overall_similarity_score = overall_similarity_score * num_functions_similarity
                function_similarity_scores.append(overall_similarity_score)

        final_similarity_score = sum(function_similarity_scores) / len(function_similarity_scores)
        return final_similarity_score
    
    else:

        final_score = 0
        return final_score

def string_sim(file1_content, file2_content):
    
    file1_strings = re.findall(r'"([^"]*)"', file1_content) + re.findall(r"'([^']*)'", file1_content)
    file2_strings = re.findall(r'"([^"]*)"', file2_content) + re.findall(r"'([^']*)'", file2_content)

    # Calculate similarity using SequenceMatcher from difflib
    similarity_score = difflib.SequenceMatcher(None, file1_strings, file2_strings).ratio()

    return similarity_score


def variables_sim(file1_content, file2_content):
    file1_ast = ast.parse(file1_content)
    file2_ast = ast.parse(file2_content)

    file1_variables = set()
    for node in ast.walk(file1_ast):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    file1_variables.add(target.id)
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                file1_variables.add(arg.arg)

    file2_variables = set()
    for node in ast.walk(file2_ast):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    file2_variables.add(target.id)
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                file2_variables.add(arg.arg)

    if not file1_variables or not file2_variables:
        return 0.0

    # Calculate Jaccard similarity between file1_variables and file2_variables
    similarity = len(file1_variables.intersection(file2_variables)) / len(file1_variables.union(file2_variables))

    return similarity

def loop_sim(file1_content, file2_content):
    file1_ast = ast.parse(file1_content)
    file2_ast = ast.parse(file2_content)

    file1_loops = ""
    file2_loops = ""

    for node in ast.walk(file1_ast):
        if isinstance(node, (ast.For, ast.While)):
            file1_loops +=  file1_content.splitlines()[node.lineno-1].strip()
            
    for node in ast.walk(file2_ast):
        if isinstance(node, (ast.For, ast.While)):
            file2_loops += file2_content.splitlines()[node.lineno-1].strip()

    if file1_loops == "" or file2_loops == "":
        return 0
    
    similarity_score = 1 - levenshtein_distance(file1_loops, file2_loops)

    # loop_similarity_score /= len(file1_loops) if len(file1_loops) > 0 else 1

    return similarity_score

def levenshtein_distance(seq1, seq2):

    len1, len2 = len(seq1), len(seq2)
    # Initialize the distance matrix
    dp = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]
    # Set the initial values
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    # Calculate the Levenshtein distance
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    # Normalize the distance
    max_len = max(len1, len2)
    normalized_distance = dp[len1][len2] / max_len if max_len > 0 else 0.0
    return normalized_distance

def encode_vertical_structure(lines):
    encoded_string = ""

    for line in lines:
        line = line.strip('\n')
        if not line.strip():
            encoded_string += "0"
            continue

        if line.strip().startswith("#"):
            encoded_string += "#"
            continue
        encoded_string += "1"
    return encoded_string

def extract_comments(lines):
    comments = []
    for line in lines:
        comment = re.findall(r'#\s*(.*)', line.strip())
        if comment:
            comments.append(comment[0])
    return comments

def preprocess_comments(comments):
    return [re.sub(r'[^\w\s]', '', comment.lower()) for comment in comments]

def comment_sim(lines1, lines2):
    comments1 = extract_comments(lines1)
    comments2 = extract_comments(lines2)

    if not comments1 or not comments2:
        return 0

    preprocessed_comments1 = preprocess_comments(comments1)
    preprocessed_comments2 = preprocess_comments(comments2)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_comments1 + preprocessed_comments2)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    n = len(comments1)
    m = len(comments2)
    max_similarities = []

    for i in range(n):
        max_similarity = 0
        for j in range(n, n + m):
            max_similarity = max(max_similarity, similarity_matrix[i, j])
        max_similarities.append(max_similarity)

    if not max_similarities:
        return 0

    avg_max_similarity = sum(max_similarities) / n

    return avg_max_similarity

def import_sim(lines1, lines2, import_count):
    imports1 = set()
    for line in lines1:
        match = re.search(r'import (\S+)|from (\S+) import', line)
        if match:
            library = match.group(1) if match.group(1) else match.group(2)
            imports1.add(library)

    imports2 = set()
    for line in lines2:
        match = re.search(r'import (\S+)|from (\S+) import', line)
        if match:
            library = match.group(1) if match.group(1) else match.group(2)
            imports2.add(library)
    common_imports = imports1.intersection(imports2)
    similarity_import = sum(1/import_count[library] for library in common_imports)

    return similarity_import

def similarity_vertical(lines1, lines2):
    seq1 = encode_vertical_structure(lines1)
    seq2 = encode_vertical_structure(lines2)
    similarity_vertical = 1 - levenshtein_distance(seq1, seq2)
    return similarity_vertical

def moss_func(moss_file):
    moss_file = pd.read_excel(moss_file)

    # precision_moss and recall_moss Calc - Direct and Indirect
    true_positives_moss = moss_file[(moss_file['Range'] >= '50-60')]['Direct'].sum() + moss_file[(moss_file['Range'] >= '50-60')]['Indirect'].sum()
    false_positives_moss = moss_file[(moss_file['Range'] >= '50-60')]['Original'].sum()
    false_negatives_moss = moss_file[(moss_file['Range'] <= '40-50')]['Direct'].sum() + moss_file[(moss_file['Range'] <= '40-50')]['Indirect'].sum()

    # precision_moss and recall_moss Calc - Originals
    true_positives_moss_orig = moss_file[(moss_file['Range'] <= '40-50')]['Original'].sum() 
    false_positives_moss_orig = moss_file[(moss_file['Range'] <= '40-50')]['Direct'].sum() + moss_file[(moss_file['Range'] <= '40-50')]['Indirect'].sum()
    false_negatives_moss_orig = moss_file[(moss_file['Range'] >= '50-60')]['Original'].sum()

    precision_moss = true_positives_moss / (true_positives_moss + false_positives_moss)
    recall_moss = true_positives_moss / (true_positives_moss + false_negatives_moss)
    f1_moss = (2*precision_moss*recall_moss)/(precision_moss+recall_moss)
    precision_moss_orig = true_positives_moss_orig / (true_positives_moss_orig + false_positives_moss_orig)
    recall_moss_orig = true_positives_moss_orig / (true_positives_moss_orig + false_negatives_moss_orig)
    f1_moss_orig = (2*precision_moss_orig*recall_moss_orig)/(precision_moss_orig + recall_moss_orig)

    moss_rows = [
    {'Range': 'Precision', 'Direct': precision_moss, 'Indirect' : '', 'Original': precision_moss_orig},
    {'Range': 'Recall', 'Direct': recall_moss, 'Indirect' : '', 'Original': recall_moss_orig},
    {'Range': 'F1 Score', 'Direct': f1_moss, 'Indirect' : '', 'Original': f1_moss_orig},
    ]

    moss_file = moss_file.append(moss_rows, ignore_index=True)
    return moss_file

def jplag_func(jplag_file):

    jplag_file = pd.read_excel(jplag_file)

    # precision_jplag and recall_jplag Calc - Direct and Indirect
    true_positives_jplag = jplag_file[(jplag_file['Range'] >= '50-60')]['Direct'].sum() + jplag_file[(jplag_file['Range'] >= '50-60')]['Indirect'].sum()
    false_positives_jplag = jplag_file[(jplag_file['Range'] >= '50-60')]['Original'].sum()
    false_negatives_jplag = jplag_file[(jplag_file['Range'] <= '40-50')]['Direct'].sum() + jplag_file[(jplag_file['Range'] <= '40-50')]['Indirect'].sum()

    # precision_jplag and recall_jplag Calc - Originals
    true_positives_jplag_orig = jplag_file[(jplag_file['Range'] <= '40-50')]['Original'].sum() 
    false_positives_jplag_orig = jplag_file[(jplag_file['Range'] <= '40-50')]['Direct'].sum() + jplag_file[(jplag_file['Range'] <= '40-50')]['Indirect'].sum()
    false_negatives_jplag_orig = jplag_file[(jplag_file['Range'] >= '50-60')]['Original'].sum()

    precision_jplag = true_positives_jplag / (true_positives_jplag + false_positives_jplag)
    recall_jplag = true_positives_jplag / (true_positives_jplag + false_negatives_jplag)
    f1_jplag = (2*precision_jplag*recall_jplag)/(precision_jplag+recall_jplag)
    precision_jplag_orig = true_positives_jplag_orig / (true_positives_jplag_orig + false_positives_jplag_orig)
    recall_jplag_orig = true_positives_jplag_orig / (true_positives_jplag_orig + false_negatives_jplag_orig)
    f1_jplag_orig = (2*precision_jplag_orig*recall_jplag_orig)/(precision_jplag_orig + recall_jplag_orig)

    jplag_rows = [
    {'Range': 'Precision', 'Direct': precision_jplag, 'Indirect' : '', 'Original': precision_jplag_orig},
    {'Range': 'Recall', 'Direct': recall_jplag, 'Indirect' : '', 'Original': recall_jplag_orig},
    {'Range': 'F1 Score', 'Direct': f1_jplag, 'Indirect' : '', 'Original': f1_jplag_orig},
    ]

    jplag_file = jplag_file.append(jplag_rows, ignore_index=True)
    return jplag_file

def compare_files(file1, file2, import_count):
    file1_content = read_file(file1)
    file2_content = read_file(file2)
    
    comment_similarity = comment_sim(file1_content, file2_content)    
    # import_similarity = import_sim(file1_content, file2_content, import_count)*0
    vertical_similarity = similarity_vertical(file1_content, file2_content)

    # Join the lines in each file into a single string
    file1_content = ''.join(file1_content)
    file2_content = ''.join(file2_content)

    function_similarity = function_sim(file1_content, file2_content)
    ast_similarity = ast_sim(file1_content, file2_content) 
    string_similarity = string_sim(file1_content, file2_content)
    variables_similarity = variables_sim(file1_content, file2_content)
    loop_similarity = loop_sim(file1_content, file2_content)

    overall_similarity_score = (comment_similarity + function_similarity + ast_similarity + string_similarity + variables_similarity + loop_similarity + vertical_similarity)/8*1000

    return overall_similarity_score,comment_similarity, vertical_similarity, function_similarity, ast_similarity, string_similarity, variables_similarity, loop_similarity

def compare_files_in_folder(folder_path):
    files = os.listdir(folder_path)
    similarity_scores = []

    python_files = glob2.glob(f"{folder_path}/*.py")    
    import_count = defaultdict(int)
    for file1 in python_files:
            with open(file1, 'r') as f1:
                content = f1.read()
                imports = re.findall(r'import (\S+)|from (\S+) import', content)
                for imp in imports:
                    library = imp[0] if imp[0] else imp[1]
                    import_count[library] += 1
    
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            if i < j:
                print(files[i], files[j])
                file1 = os.path.join(folder_path, files[i])
                file2 = os.path.join(folder_path, files[j])
                similarity_score, comment_similarity,vertical_similarity, function_similarity, ast_similarity, string_similarity, variables_similarity, loop_similarity = compare_files(file1, file2, import_count)
                
                if similarity_score > 90:
                    similarity_score_1 = '90-100'
                elif similarity_score > 80:
                    similarity_score_1 = '80-90'
                elif similarity_score > 70:
                    similarity_score_1 = '70-80'
                elif similarity_score > 60:
                    similarity_score_1 = '60-70'
                elif similarity_score > 50:
                    similarity_score_1 = '50-60'
                elif similarity_score > 40:
                    similarity_score_1 = '40-50'
                elif similarity_score > 30:
                    similarity_score_1 = '30-40'
                elif similarity_score > 20:
                    similarity_score_1 = '20-30'
                elif similarity_score > 10:
                    similarity_score_1 = '10-20'
                else:
                    similarity_score_1 = '0-10'


                if files[i][0] == files[j][0]:
                    if len(files[i]) == 6:
                        similarity_scores.append((files[i], files[j], 'Indirect', similarity_score,  similarity_score_1, comment_similarity, vertical_similarity, function_similarity, ast_similarity, string_similarity, variables_similarity, loop_similarity))

                    else:
                        similarity_scores.append((files[i], files[j], 'Direct', similarity_score, similarity_score_1, comment_similarity, vertical_similarity, function_similarity, ast_similarity, string_similarity, variables_similarity, loop_similarity))

                else:
                    similarity_scores.append((files[i], files[j], 'Original', similarity_score, similarity_score_1, comment_similarity, vertical_similarity, function_similarity, ast_similarity, string_similarity, variables_similarity, loop_similarity))


    # DataFrame 1: to store the similarity scores and inidvidual Similarity Scores
    df = pd.DataFrame(similarity_scores, columns=['File1', 'File2', 'Plg/Original', 'Similarity Score', 'Range', 'Comment', 'Vertical', 'Function', 'AST', 'String', 'Variable', 'Loop'])

    # Dataframe 2: Scores in ranges- Summary
    final_graph = []
    ranges = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    for r in ranges:
        count1 = 0
        count2 = 0
        count3 = 0
        for x in range(len(df['Range'])):
            if df['Range'][x] == r:
                if df['Plg/Original'][x] == 'Direct':
                    count1 += 1
                elif df['Plg/Original'][x] == 'Indirect':
                    count2 += 1
                else:
                    count3 += 1
        final_graph.append([r, count1, count2, count3])
    new = pd.DataFrame(final_graph, columns=['Range', 'Direct', 'Indirect', 'Original'])
        
    # Precision and Recall Calc - Direct and Indirect
    true_positives = new[(new['Range'] >= '50-60')]['Direct'].sum() + new[(new['Range'] >= '50-60')]['Indirect'].sum()
    false_positives = new[(new['Range'] >= '50-60')]['Original'].sum()
    false_negatives = new[(new['Range'] <= '40-50')]['Direct'].sum() + new[(new['Range'] <= '40-50')]['Indirect'].sum()

    # Precision and Recall Calc - Originals
    true_positives_orig = new[(new['Range'] <= '40-50')]['Original'].sum() 
    false_positives_orig = new[(new['Range'] <= '40-50')]['Direct'].sum() + new[(new['Range'] <= '40-50')]['Indirect'].sum()
    false_negatives_orig = new[(new['Range'] >= '50-60')]['Original'].sum()

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = (2*precision*recall)/(precision+recall)
    precision_orig = true_positives_orig / (true_positives_orig + false_positives_orig)
    recall_orig = true_positives_orig / (true_positives_orig + false_negatives_orig)
    f1_orig = (2*precision_orig*recall_orig)/(precision_orig + recall_orig)

    new_rows = [
    {'Range': 'Precision', 'Direct': precision, 'Indirect' : '', 'Original': precision_orig},
    {'Range': 'Recall', 'Direct': recall, 'Indirect' : '', 'Original': recall_orig},
    {'Range': 'F1 Score', 'Direct': f1, 'Indirect' : '', 'Original': f1_orig},
    ]

    new = new.append(new_rows, ignore_index=True)

    # Dataframe 3- with averages
    df = df.drop(df.columns[4], axis=1)
    selected_columns = df.columns[3:]
    averages_df = pd.DataFrame(index=['Direct', 'Indirect', 'Original'], columns=selected_columns)

    for category in ['Direct', 'Indirect', 'Original']:
        filtered_df = df[df['Plg/Original'] == category]
        averages = filtered_df.iloc[:, 3:].mean()
        averages_df.loc[category] = averages

    # Dataframe 4 and 5- moss and jplag
    moss_df = moss_func(moss_file)
    jplag_df = jplag_func(jplag_file)

    # Excel Stuff
    filename = "{}_final.xlsx".format(question)
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    new.to_excel(writer, sheet_name='Proposed_Summary', index=False)
    df.to_excel(writer, sheet_name='Proposed_All', index=False)
    averages_df.to_excel(writer, sheet_name='Proposed_Averages', index=True)
    moss_df.to_excel(writer, sheet_name='MOSS_Summary', index=False)
    jplag_df.to_excel(writer, sheet_name='JPlag_Summary', index=False)
    writer.save()

question = input("Enter Question Character: ")
# folder_path = "C:\\Users\\nandi\\Desktop\\plgchecker\\{}".format(question)
moss_file = "moss_{}.xlsx".format(question)
jplag_file = "jplag_{}.xlsx".format(question)

print(moss_func(moss_file))
print(jplag_func(jplag_file))

# for file_name in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, file_name)
#     if os.path.isfile(file_path):
#         with open(file_path, 'r') as file:
#             file_content = file.read()
# compare_files_in_folder(folder_path)
