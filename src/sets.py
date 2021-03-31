import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def make_val_set(mrna_df, full_df):
    """[Takes rna formatted dataframe and combines with clinical dataset mortality column]

    Args:
        mrna_df ([pandas dataframe): [RNA zscored format pandas dataframe]
        full_df ([full clinical dataset]): [full metabric clinical dataframe]

    Returns:
        [Train set, Final validation set]: [Train set is traditional, test train split group, validation set is sequestered data not to be mixed or seen]
    """
    death_from_dict = {
    'Living':0
    ,'Died of Other Causes':0
    ,'Died of Disease':1
    }
    full_df.replace(death_from_dict, inplace =True)
    mrna_df['death_from_cancer'] = full_df.death_from_cancer
    mrna_df.death_from_cancer.fillna(0, inplace = True)
    mrna_df = shuffle(mrna_df)
    Validation_set = mrna_df[:200]
    mrna1 = mrna_df[200:]
    # Validation_set.to_csv(r'../data/validation_wednesday_set.csv', index = False)
    # mrna1.to_csv(r'../data/rna_df_wednesday.csv', index = False)
    
    return mrna1, Validation_set

def mutant_list_maker(df):
    """[makes separate lst of columns that are mutants]

    Args:
        df ([Pandas dataframe]): [columns contain _mut in title if genes are mutated]

    Returns:
        [list]: [list of columns that are mutants]
    """
    lst = []
    for col in df.columns:
        if '_mut' in col:
            lst.append(col)
    return lst

# def non_numerical_column_separator(df):
#     """[returns list of columns that a]

#     Args:
#         df ([type]): [description]

#     Returns:
#         [type]: [description]
#     """
#     lst = []
#     for col in df.columns:
#         if col[0]type(i) =='str':
#             lst.append(i)
#     return lst

def change_cols_to_floats(dataframe,lst):
    """[Takes dataframe and list and turns columns of dataframe in list to floats]

    Args:
        dataframe ([Pandas Dataframe]): [pandas METABRIC dataframe]
        lst ([list]): [list of columns to convert to floats]

    Returns:
        [pandas dataframe]: [converted to floats dataframe]
    """
    
    for i in lst:

        dataframe[i] = dataframe[i].astype(float)
    return dataframe

def make_number_cols(df):
    """[Converts numerical catagories to floats because downloaded format is string]

    Args:
        df ([pandas df]): [metabric data]
       

    Returns:
        [list]: [list of columns to be converted to floats]
    """
    lst_non_number = ['type_of_breast_surgery', 'cancer_type', 'cancer_type_detailed', 'cellularity', 'pam50_+_claudin-low_subtype', 'er_status_measured_by_ihc', 
    'er_status', 'her2_status_measured_by_snp6', 'her2_status','tumor_other_histologic_subtype', 'inferred_menopausal_state', 'integrative_cluster','primary_tumor_laterality', 
    'oncotree_code', 'pr_status', '3-gene_classifier_subtype', 'death_from_cancer']
    lst_mutants =mutant_list_maker(df)
    lst_total_number = lst_non_number +lst_mutants
    lst = []
    for col in df.columns:
        if col not in lst_total_number:
            lst.append(col)

    return lst

def convert_surgury(df):
    """[convert surgury column to numerical]

    Args:
        df ([pandas]): [dataframe]

    Returns:
        [dataframe]: [converted surgury column 1 == masectomy, 0 == Brest Conservering]
    """
    surgury_dct = {
        'MASTECTOMY':1
        ,'BREAST CONSERVING':0
    }
    df['type_of_breast_surgery'].replace(surgury_dct, inplace = True)
    return df

def convert_er_status(df):
    """[convert column to numerical]

    Args:
        df ([pandas]): [dataframe]

    Returns:
        [dataframe]: [converted estrogen receptor column to numerical  1 == Positive, 0 == Negative]
    """
    er_status_dct = {
        'Positive':1
        ,'Negative':0
    }
    df['er_status'].replace(er_status_dct, inplace = True)
    return df

def convert_er_status_measured_by_ihc(df):
     """[convert column to numerical]

    Args:
        df ([pandas]): [dataframe]

    Returns:
        [dataframe]: [converted estrogen receptor column to numerical  1 == Positive, 0 == Negative]
    """
    convert_er_status_measured_by_ihc_dct = {
        'Positve':1
        ,'Negative':0
    }
    df['er_status_measured_by_ihc'].replace(convert_er_status_measured_by_ihc_dct, inplace = True)
    return df

def convert_inferred_menopausal_state(df):
     """[convert column to numerical]

    Args:
        df ([pandas]): [dataframe]

    Returns:
        [dataframe]: [converted menopausal column to numerical  1 == Post, 0 == Pre]
    """
    convert_inferred_menopausal_state_dct = {
        'Post':1
        ,'Pre':0
    }
    df['inferred_menopausal_state'].replace(convert_inferred_menopausal_state_dct, inplace = True)
    return df

def convert_primary_tumor_laterality(df):
    """[convert column to numerical]

    Args:
        df ([pandas]): [dataframe]

    Returns:
        [dataframe]: [converted tumor laterality column to numerical  1 == Right, 0 == Left]
    """
    convert_primary_tumor_laterality_dct = {
        'Right':1
        ,'Left':0
    }
    df['primary_tumor_laterality'].replace(convert_primary_tumor_laterality_dct, inplace = True)
    return df

def convert_pr_status(df):
     """[convert column to numerical]

    Args:
        df ([pandas]): [dataframe]

    Returns:
        [dataframe]: [converted PR column to numerical  1 == Postiive, 0 == Negative]
    """
    convert_pr_status_dct = {
        'Positive':1
        ,'Negative':0
    }
    df['pr_status'].replace(convert_pr_status_dct, inplace = True)
    return df


def column_mutant_value_counts(df):
    for col in df.columns:
        print(f'name: {col} \n{df[col].value_counts()}')
def column_print(df):
    for col in df.columns:
        print(f'name: {col}')



if __name__ == '__main__':
    mrna_df =pd.read_csv('/Users/cp/Documents/dsi/capstone2/capstone2/data/capstone2.mrn_df2.csv')
    df = pd.read_csv('/Users/cp/Documents/dsi/capstone2/capstone2/data/METABRIC_RNA_Mutation.csv')

    print(make_val_set(mrna_df, df))


