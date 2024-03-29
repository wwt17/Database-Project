import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mysql.connector
from mysql.connector import errorcode
import tqdm
from models import MLP
from server_interaction import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


bool_tuples = [('No', 'Yes')]
int_tuples = [
    ('Poor', 'Fair', 'Good', 'Very good', 'Excellent'),
    ('18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'),
]
#gender_tuple = ('Female', 'Male')
#race_tuple = ('White', 'Black', 'Hispanic', 'Asian', 'American Indian/Alaskan Native', 'Other')
data_names_of_kind = {}


def get_data_names_of_kind(df):
    data_names_of_kind = defaultdict(list)
    for column_label in df.columns:
        column = df[column_label]
        if column.dtype == 'object':
            categorical_column = column.astype('category')
            categories = categorical_column.cat.categories
            if len(categories) <= 20:
                categories_tuple = tuple(categories)
                data_names_of_kind[categories_tuple].append(column_label)
    return data_names_of_kind


def convert_dtype(data, data_name=None):
    if isinstance(data, pd.DataFrame):
        for column_label in df.columns:
            df[column_label] = convert_dtype(df[column_label], column_label)
        return df
    elif isinstance(data, pd.Series):
        for bool_tuple in bool_tuples:
            if data_name in data_names_of_kind[tuple(sorted(bool_tuple))]:
                return data.map({cat: flag for flag, cat in zip([False, True], bool_tuple)})
        for int_tuple in int_tuples:
            if data_name in data_names_of_kind[tuple(sorted(int_tuple))]:
                return data.map({cat: idx for idx, cat in enumerate(int_tuple)}).astype('int16')
        for kind_tuple, data_names in data_names_of_kind.items():
            if data_name in data_names:
                return data.astype('category')
        if all(isinstance(d, str) for d in data):
            data = data.astype('string')
    else:
        raise NotImplementedError
    return data


numpy_to_sql_data_type = {
    'bool': 'BOOL',
    'int16': 'SMALLINT',
    'int32': 'INT',
    'int64': 'BIGINT',
    'float32': 'FLOAT',
    'float64': 'DOUBLE',
    'string': 'VARCHAR(30)',
}


def convert_numpy_dtype_to_python_type(x):
    try:
        x = x.item()
    except:
        pass
    return x


def row_to_mysql_tuple(row, cursor, table_name):
    """Insert a row/tuple into the table.
    """
    cmd = f"INSERT INTO {table_name} ({', '.join(row.index)}) VALUES ({', '.join(['%s'] * len(row))})"
    tup = tuple(convert_numpy_dtype_to_python_type(x) for x in row)
    return cursor.execute(cmd, tup)


def dataframe_to_mysql_table(df, cursor, table_name, drop_table_if_exists=True):
    if drop_table_if_exists:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")

    # create table
    column_specs = ["id serial PRIMARY KEY"]
    for column_label, dtype in df.dtypes.items():
        if dtype == 'category':
            data_type = f"ENUM({', '.join(map(repr, dtype.categories))})"
        else:
            data_type = numpy_to_sql_data_type[str(dtype)]
        column_spec = f"{column_label} {data_type}"
        column_specs.append(column_spec)
    cmd = f"CREATE TABLE {table_name} ({', '.join(column_specs)});"
    print(cmd)
    try:
        cursor.execute(cmd)
    except mysql.connector.Error as err:
        pass

    # insert tuples
    cmd = f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({', '.join(['%s'] * len(df.columns))})"
    seq = list(df.itertuples(index=False, name=f'{table_name}_tuple'))
    print(f'uploading {len(df)} tuples...')
    cursor.executemany(cmd, seq)


def rows_to_dataframe(rows):
    return pd.DataFrame(rows)


def mysql_table_to_dataframe(cursor, table_name, dtypes):
    print(f'Retrieving {table_name}...')
    cmd = f"SELECT * FROM {table_name};"
    try:
        cursor.execute(cmd)
    except mysql.connector.Error as err:
        print(err)
    else:
        rows = cursor.fetchall()
        df = rows_to_dataframe(rows)
        df.drop(0, axis=1, inplace=True)
        df.rename(dict(zip(df.columns, dtypes.index)), axis=1, inplace=True)
        for column_label, dtype in dtypes.items():
            df[column_label] = df[column_label].astype(dtype)
        return df


def normalize(ratio):
    s = sum(ratio)
    return [x / s for x in ratio]


def input_row(dtypes, skip_labels=[]):
    row = []
    for column_label, dtype in dtypes.items():
        if column_label not in skip_labels:
            if dtype == 'category':
                dtype_prompt = 'category: ' + '/'.join(map(str, dtype.categories))
            elif dtype == 'bool':
                dtype_prompt = 'bool: T(rue)/Y(es)/1/F(alse)/N(o)/0'
            else:
                for int_tuple in int_tuples:
                    if column_label in data_names_of_kind[tuple(sorted(int_tuple))]:
                        dtype_prompt = 'category: ' + ' / '.join(f'{idx}: {cat}' for idx, cat in enumerate(int_tuple))
                        break
                else:
                    dtype_prompt = str(dtype)
            while True:
                inp = input(f'{column_label} ({dtype_prompt}): ').strip()
                if dtype == 'bool':
                    inp = inp.lower()
                    if inp in ['true', 'yes', 't', 'y', '1']:
                        inp = True
                    elif inp in ['false', 'no', 'f', 'n', '0']:
                        inp = False
                    else:
                        print('Incorrect type of value.')
                        continue
                try:
                    value = pd.Series([inp], index=[column_label]).astype(dtype)
                except ValueError:
                    print('Incorrect type of value.')
                else:
                    break
            row.append(value)

    row = pd.concat(row)
    return row


def get_input_output_dim(dtypes, predicted_labels):
    input_dim = 0
    output_dim = 0
    for column_label, dtype in dtypes.items():
        if dtype == 'category':
            dim = len(dtype.categories)
        else:
            dim = 1
        if column_label in predicted_labels:
            output_dim += dim
        else:
            input_dim += dim
    return input_dim, output_dim


def get_input_output_features(row, dtypes, predicted_labels):
    input_features = []
    output_features = []
    for column_label, value in row.items():
        dtype = dtypes[column_label]
        if dtype == 'category':
            feature = [int(value == cat) for cat in dtype.categories]
        else:
            feature = [int(value) if isinstance(value, bool) else value]
        if column_label in predicted_labels:
            output_features += feature
        else:
            input_features += feature
    input_features = torch.tensor(input_features, dtype=torch.float)
    output_features = torch.tensor(output_features, dtype=torch.float)
    return input_features, output_features


class DataFrameDataset(Dataset):
    def __init__(self, df, predicted_labels):
        self.df = df
        self.predicted_labels = predicted_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return get_input_output_features(
            self.df.iloc[idx], self.df.dtypes, self.predicted_labels)


def build_model(dtypes, predicted_labels, args):
    input_dim, output_dim = get_input_output_dim(dtypes, predicted_labels)
    if args.hidden_dim is None:
        hidden_dim = input_dim
    else:
        hidden_dim = args.hidden_dim
    return MLP(input_dim, output_dim, hidden_dim, act=args.act, dropout=args.dropout)


def load_model(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)


def save_model(model, path):
    state_dict = model.state_dict()
    torch.save(state_dict, path)


class PredTargetCounter:
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self.cnt = np.zeros((n_classes, n_classes), dtype=np.int32)

    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            self.cnt[pred, target] += 1

    @property
    def accuracy(self):
        return np.trace(self.cnt) / np.sum(self.cnt)

    @property
    def precision(self):
        assert self.n_classes == 2
        return self.cnt[1, 1] / self.cnt[1].sum()

    @property
    def recall(self):
        assert self.n_classes == 2
        return self.cnt[1, 1] / self.cnt[:, 1].sum()

    @property
    def f1(self):
        return 2 / (1 / self.precision + 1 / self.recall)


def train_epoch(model, optimizer, loss_fn, dataset, batch_size, num_workers=4):
    """Train model on dataset for an epoch.
    Return: Mean train loss.
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers)
    model.train()

    n_batches = 0
    sum_loss = 0.
    sum_cnt = PredTargetCounter()
    for inputs, targets in tqdm.tqdm(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = loss_fn(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n_batches += 1
        sum_loss += loss.item()
        sum_cnt.update(
            (logits > 0).long().cpu().numpy(),
            targets.long().cpu().numpy())

    mean_loss = sum_loss / n_batches
    print(f"train loss: {mean_loss:.3f}")
    print(f"train acc: {sum_cnt.accuracy:.3%} prec: {sum_cnt.precision:.3%} recall: {sum_cnt.recall:.3%} f1: {sum_cnt.f1:.3%}")
    return mean_loss


def evaluate(model, loss_fn, dataset, batch_size, split='eval', num_workers=4):
    """Evaluate model on dataset.
    Return: Accuracy.
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers)
    model.eval()

    n_batches = 0
    sum_loss = 0.
    sum_cnt = PredTargetCounter()
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = loss_fn(logits, targets)

            n_batches += 1
            sum_loss += loss.item()
            sum_cnt.update(
                (logits > 0).long().cpu().numpy(),
                targets.long().cpu().numpy())

    mean_loss = sum_loss / n_batches
    print(f"{split} loss: {mean_loss:.3f}")
    print(f"{split} acc: {sum_cnt.accuracy:.3%} prec: {sum_cnt.precision:.3%} recall: {sum_cnt.recall:.3%} f1: {sum_cnt.f1:.3%}")
    return sum_cnt.f1


def train_model(
    model,
    dataset,
    predicted_labels,
    args,
    save_path=None,
):
    """Train model on dataset.
    Assume predicted_labels are bool.
    """
    print('Training model...')

    datasets = {
        split: DataFrameDataset(df, predicted_labels)
        for split, df in dataset.items()
    }
    for split, df in dataset.items():
        print(f'predicted label distribution of {split} split:')
        for predicted_label in predicted_labels:
            print(f'{predicted_label} distribution:')
            print(df[predicted_label].value_counts(normalize=True))

    assert len(predicted_labels) == 1, "Only support predicting one label"
    predicted_label = predicted_labels[0]
    predicted_label_counts = dataset['train'][predicted_label].value_counts()
    pos_weight = predicted_label_counts[False] / predicted_label_counts[True]
    pos_weight = torch.tensor(pos_weight)
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f'Epoch #{0}:')
    best_eval_result = evaluate(model, loss_fn, datasets['val'], args.batch_size, split='val', num_workers=args.num_workers)
    best_epoch = 0
    if save_path is not None:
        save_model(model, save_path)

    for n_epoch in range(args.max_n_epochs):
        print(f'Epoch #{n_epoch+1}:')
        train_epoch(model, optimizer, loss_fn, datasets['train'], args.batch_size, num_workers=args.num_workers)
        eval_result = evaluate(model, loss_fn, datasets['val'], args.batch_size, split='val', num_workers=args.num_workers)
        if eval_result > best_eval_result or np.isnan(best_eval_result):
            print('update best eval result')
            best_eval_result = eval_result
            best_epoch = n_epoch + 1
            if save_path is not None:
                save_model(model, save_path)

    load_model(model, save_path)


def train_machine_learning_model(model, df, predicted_labels, args, model_blob_name='model.pt'):
    s = 0.
    split_points = [0]
    for x in args.split_ratio:
        s += x
        split_points.append(int(len(df) * s))
    split_points[-1] = len(df)
    data_splits = [df[split_points[i]:split_points[i+1]] for i in range(len(args.split_ratio))]
    dataset = {split: data_split for split, data_split in zip(['train', 'val', 'test'], data_splits)}

    train_model(model, dataset, predicted_labels, args, save_path=args.save_model)

    print('uploading model...')
    container_client = get_blob_container_client()
    while True:
        with open(args.save_model, 'rb') as data:
            try:
                container_client.upload_blob(model_blob_name, data)
            except azure.core.exceptions.ResourceExistsError:
                container_client.delete_blob(model_blob_name)
            else:
                print('uploaded.')
                break


def init(df, conn, cursor, table_name, model, predicted_labels, args):
    print('Initializing:')

    # upload df to mysql database as a table
    dataframe_to_mysql_table(df, cursor, table_name)
    conn.commit()

    # train the machine learning model
    if args.load_model is not None:
        load_model(model, args.load_model)
    train_machine_learning_model(model, df, predicted_labels, args)
    n_accu_updates = 0


if __name__ == '__main__':
    print("Welcome to the interactive client of Wentao Wang's Database project!")

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--csv_file', type=Path, default=Path('data.csv'),
        help='The dataset file to use. '
             'This is a .csv file containing tuples of the dataset. '
             'Default to data.csv.'
    )
    argparser.add_argument(
        '--n_init_tuples', type=int, default=5000,
        help='Number of initial tuples in the beginning of the file. '
             'We do not use the full dataset because it is very slow to upload '
             'many tuples. '
             'Default to 5000.'
    )
    ml_group = argparser.add_argument_group(title='Machine Learning Model')
    ml_group.add_argument(
        '--split_ratio', nargs=3, type=float, default=[3, 1, 1],
        help='The ratio of train/val/test splits. '
             'Represented by numbers separated by space. '
             'Default to 3 1 1.'
    )
    ml_group.add_argument(
        '--hidden_dim', type=int, default=512,
        help='Number of dimensions of the hidden state. '
             'Default to 512.'
    )
    ml_group.add_argument(
        '--act', choices=['relu', 'tanh'], default='relu',
        help='Activation function. '
             'Default to relu.'
    )
    ml_group.add_argument(
        '--dropout', type=float, default=0.5,
        help='Dropout rate applied in the model. '
             'Can be any number between 0 (no dropout) and 1. '
             'Default to 0.5.'
    )
    ml_group.add_argument(
        '--batch_size', type=int, default=1024,
        help='Batch size to train and evaluate the model. '
             'Default to 1024.'
    )
    ml_group.add_argument(
        '--lr', type=float, default=1e-1,
        help='Learning rate. '
             'Default to 0.1.'
    )
    ml_group.add_argument(
        '--weight_decay', type=float, default=0.01,
        help='Weight decay. '
             'Default to 0.01.'
    )
    ml_group.add_argument(
        '--max_n_epochs', type=int, default=20,
        help='Max number of training epochs. '
             'Set to 0 if you do not want to train the model. '
             'Default to 100.'
    )
    ml_group.add_argument(
        '--num_workers', type=int, default=16,
        help='Number of dataloader workers. '
             'Default to 16.'
    )
    ml_group.add_argument(
        '--load_model', type=Path, default=None,
        help='Load model parameters from this path. '
             'Default to not loading model parameters (random initalization).'
    )
    ml_group.add_argument(
        '--save_model', type=Path, default=Path('model.pt'),
        help='Save model parameters to this path. '
             'Default to model.pt.'
    )
    ml_group.add_argument(
        '--seed', type=int, default=0,
        help='Random seed for reproducibility. '
             'Default to 0.'
    )
    ml_group.add_argument(
        '--device', type=str, default=None,
        help='torch.device to use. '
             'Default to detecting the environment '
             '(use cuda when cuda is available).'
    )
    argparser.add_argument(
        '--retrain_interval', type=int, default=1000,
        help='Retrain after adding/deleting how many tuples. '
             'Default to a large value 1000 since retraining is time-consuming.'
    )
    args = argparser.parse_args()
    if args.device is not None:
        device = torch.device(args.device)
    torch.manual_seed(args.seed)
    args.split_ratio = normalize(args.split_ratio)

    csv_file = args.csv_file
    assert csv_file.suffix == ".csv", f"{csv_file} is not a csv file"
    table_name = csv_file.stem

    df = pd.read_csv(csv_file)
    data_names_of_kind = get_data_names_of_kind(df)
    df = convert_dtype(df)
    predicted_labels = df.columns[:1]

    try:
        conn = get_mysql_connector()
    except mysql.connector.Error as err:
        raise
    else:
        cursor = conn.cursor()

    model = build_model(df.dtypes, predicted_labels, args=args)
    model = model.to(device)

    def _retrain():
        cur_df = mysql_table_to_dataframe(cursor, table_name, df.dtypes)
        train_machine_learning_model(model, cur_df, predicted_labels, args)
        n_accu_updates = 0

    if args.load_model is not None:
        load_model(model, args.load_model)
    else:
        _retrain()

    input_row_prompt = f'For convenience, you may input an integer id between {df.index[0]} and {df.index[-1]} so that the program will simply use the tuple with this id in the dataset, or you may press enter so you have to input all fields of a row: '
    n_accu_updates = 0

    while True:
        try:
            op = input('Please input your operation (query/predict/add/delete/reinit/retrain/quit): ')
        except EOFError:
            print()
            break

        if op == 'reinit':
            init(df[:args.n_init_tuples], conn, cursor, table_name, model, predicted_labels, args)

        elif op == 'query':
            print(f'You may input an SQL query on table {table_name} (end with ";"), like "SELECT * FROM {table_name};"')
            cmd = ""
            prompt = f'MySQL [{mysql_server_config["database"]}]> '
            cmd_line = ""
            while ";" not in cmd_line:
                cmd_line = input(prompt)
                if cmd:
                    cmd += "\n"
                cmd += cmd_line
                prompt = "    -> "
            try:
                cursor.execute(cmd)
            except mysql.connector.Error as err:
                print(err)
            else:
                rows = cursor.fetchall()
                result_df = rows_to_dataframe(rows)
                print(result_df)

        elif op == 'predict':
            inp = input(input_row_prompt).strip()
            try:
                idx = int(inp)
            except ValueError:  # input all fields of a row
                try:
                    row = input_row(df.dtypes, skip_labels=predicted_labels)
                except EOFError:
                    print()
                    continue
            else:  # use the tuple at idx
                row = df.loc[idx]

            print('Tuple:')
            print(row)

            inputs, targets = get_input_output_features(
                row, df.dtypes, predicted_labels)
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            preds = logits.sigmoid()

            for i, predicted_label in enumerate(predicted_labels):
                print(f'{predicted_label}:', end='')
                print(f' prediction = {preds[i].item():.3f}', end='')
                if len(targets):
                    print(f' ground-truth = {targets[i].item():.0f}', end='')
                print()

        elif op == 'add':
            inp = input(input_row_prompt).strip()
            try:
                idx = int(inp)
            except ValueError:  # input all fields of a row
                try:
                    row = input_row(df.dtypes)
                except EOFError:
                    print()
                    continue
            else:  # use the tuple at idx
                row = df.loc[idx]

            print('Tuple:')
            print(row)

            confirmation = input('Add this tuple? (Y/n): ').strip()
            if not confirmation or confirmation.lower() == 'y':
                try:
                    row_to_mysql_tuple(row, cursor, table_name)
                except mysql.connector.Error as err:
                    print(err)
                else:
                    n_accu_updates += 1

        elif op == 'delete':
            print('You may complete the SQL command by the condition of tuples to delete. For example, id = 5000')
            cmd = f'DELETE FROM {table_name} WHERE '
            inp = input(cmd)
            cmd += inp
            try:
                cursor.execute(cmd)
            except mysql.connector.Error as err:
                print(err)
            else:
                n_accu_updates += 1

        elif op == 'retrain':
            _retrain()

        elif op == 'quit':
            break

        else:
            print(f'Unknown operation: {op}')

        if n_accu_updates >= args.retrain_interval:
            _retrain()

    cursor.close()
    conn.close()
