import pandas as pd
from pymatgen.core.composition import Composition

"""
把收集到的数据集统一化学式，下面的函数来自MPEA的github代码
给生成的虚拟样本排序
"""

if __name__ == '__main__':

    def normalize_and_alphabetize_formula(formula):
        '''Normalizes composition labels. Used to enable matching / groupby on compositions.'''

        if formula:
            try:
                comp = Composition(formula)
                weights = [comp.get_atomic_fraction(ele) for ele in comp.elements]
                normalized_weights = [round(w / max(weights), 3) for w in weights]
                normalized_comp = "".join([str(x) + str(y) for x, y in zip(comp.elements, normalized_weights)])

                return Composition(normalized_comp).alphabetical_formula
            except:
                print("INVALID: ", formula)
                return None
        else:
            return None

    df = pd.read_excel("Virture_samples_100000.xlsx")
    df=pd.DataFrame(df["formula"],columns=["formula"])
    # 注意按照列名提取数据框的列是两层列表！！！
    #df_new = df[['Composition', 'HV']]
    formula_final = []
    for i in range(len(df)):
        formula=normalize_and_alphabetize_formula(df.iloc[i,0])
        formula_final.append(formula)
    df['formula_final'] = formula_final
    df=df.drop(["formula"],axis=1)
    df.columns=["formula"]
    print(df.head())
    # 如果有重复值，则保留第一个
    len1 = len(df.iloc[:, 0])
    df.drop_duplicates(keep='first', inplace=True)
    len2 = len(df.iloc[:, 0])
    print('Original set  ---> ', len1, '\ndroped duplicates   ---> ', len1 - len2)
    #droped duplicates - -->  26
    df.to_excel("Virture_samples_100000_formula_sorted.xlsx", index=False)