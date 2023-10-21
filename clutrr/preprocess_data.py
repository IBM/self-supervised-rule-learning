from os import listdir
from os.path import join
import csv
import json
from pathlib import Path
from argparse import ArgumentParser


def process_csv_row(csv_row):
    id = csv_row['id']
    text_story = csv_row['story']
    text_query = csv_row['text_query']

    entities = [p.split(':')[0] for p in csv_row['genders'].split(',')]
    graph_story = [f'{e}("{entities[s]}", "{entities[d]}")' for e, (s, d) in
                   zip(eval(csv_row['edge_types']), eval(csv_row['story_edges']))]

    src, dest = eval(csv_row['query'])
    graph_query = csv_row['target'] + f'("{src}", "{dest}")'

    genders = {p.split(':')[0]: p.split(':')[1] for p in csv_row['genders'].split(',')}
    task_name = csv_row['task_name']
    return id, text_story, text_query, graph_story, graph_query, genders, task_name


def process_csv(input_file_path, out_file_path, input_file_name):
    questions = []
    for row in csv.DictReader(open(join(input_file_path, input_file_name))):
        id, text_story, text_query, graph_story, graph_query, genders, task_name = process_csv_row(row)
        questions.append({'id': id, 'text_story': text_story, 'text_query': text_query, 'graph_story': graph_story,
                          'graph_query': graph_query, 'genders': genders, 'task_name': task_name})

    out_file_name = input_file_name.replace('csv', 'json')
    Path(out_file_path).mkdir(parents=True, exist_ok=True)

    json.dump(questions, open(join(out_file_path, out_file_name), 'w'), indent=2)
    print('Processed ' + join(input_file_path, input_file_name))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_data_path", type=str)
    parser.add_argument("--output_data_path", type=str)
    args = parser.parse_args()
    for file_name in [f for f in listdir(args.input_data_path) if f.endswith('.csv')]:
        process_csv(args.input_data_path, args.output_data_path, file_name)
