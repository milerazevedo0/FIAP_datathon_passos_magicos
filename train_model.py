import pandas as pd
from pathlib import Path
from src.train import train_pipeline

FILE = Path("BASE DE DADOS PEDE 2024 - DATATHON.xlsx")

TRAIN_SHEETS = ["PEDE2022", "PEDE2023"]
TEST_SHEETS = ["PEDE2024"]

POSSIBLE_TARGETS = ["Defas", "Defasagem"]


def identify_target(df: pd.DataFrame) -> str:
    for col in POSSIBLE_TARGETS:
        if col in df.columns:
            return col
    raise ValueError("Nenhuma coluna de target encontrada.")


def load_sheets(sheets):
    dfs = []

    for sheet in sheets:
        print(f"📄 Lendo aba: {sheet}")
        df = pd.read_excel(FILE, sheet_name=sheet)

        target_col = identify_target(df)
        df = df.rename(columns={target_col: "target_raw"})

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def main():
    print("🚀 Iniciando treinamento com validação temporal")

    df_train = load_sheets(TRAIN_SHEETS)
    df_test = load_sheets(TEST_SHEETS)

    df_train = df_train.dropna(subset=["target_raw"])
    df_test = df_test.dropna(subset=["target_raw"])

    train_pipeline(
        df_train=df_train,
        df_test=df_test,
        target_col="target_raw"
    )

    print("🎉 Treinamento temporal finalizado com sucesso!")


if __name__ == "__main__":
    main()
