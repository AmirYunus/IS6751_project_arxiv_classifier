import arxiv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def __safe_decode(text):
    """Handle encoding issues."""
    try:
        return text.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        return text.encode('utf-8').decode('windows-1252', errors='ignore')
    
def __clean_summary(summary):
    """Remove line breaks from summary."""
    return ' '.join(summary.split())

def __fetch_results(client, search, max_retries=3):
    """Handle UnexpectedEmptyPageError."""
    for _ in range(max_retries):
        try:
            return list(client.results(search))
        except arxiv.UnexpectedEmptyPageError:
            print(f"Encountered UnexpectedEmptyPageError. Retrying... ({_ + 1}/{max_retries})")
    print(f"Failed to fetch results after {max_retries} attempts.")
    return []

def __get_category_mapping():
    """Return dictionary mapping arxiv categories to parent categories. Default to 'physics' for unknown categories."""
    return {
        'astro-ph': 'physics',
        'astro-ph.co': 'physics',
        'astro-ph.ep': 'physics',
        'astro-ph.ga': 'physics',
        'astro-ph.he': 'physics',
        'astro-ph.im': 'physics',
        'astro-ph.sr': 'physics',
        'cond-mat': 'physics',
        'cond-mat.dis-nn': 'physics',
        'cond-mat.mes-hall': 'physics',
        'cond-mat.mtrl-sci': 'physics',
        'cond-mat.other': 'physics',
        'cond-mat.quant-gas': 'physics',
        'cond-mat.soft': 'physics',
        'cond-mat.stat-mech': 'physics',
        'cond-mat.str-el': 'physics',
        'cond-mat.supr-con': 'physics',
        'gr-qc': 'physics',
        'hep-ex': 'physics',
        'hep-lat': 'physics',
        'hep-ph': 'physics',
        'hep-th': 'physics',
        'math-ph': 'physics',
        'nlin': 'physics',
        'nlin.ao': 'physics',
        'nlin.cd': 'physics',
        'nlin.cg': 'physics',
        'nlin.ps': 'physics',
        'nlin.si': 'physics',
        'nucl-ex': 'physics',
        'nucl-th': 'physics',
        'physics': 'physics',
        'physics.acc-ph': 'physics',
        'physics.ao-ph': 'physics',
        'physics.app-ph': 'physics',
        'physics.atm-clus': 'physics',
        'physics.atom-ph': 'physics',
        'physics.bio-ph': 'physics',
        'physics.chem-ph': 'physics',
        'physics.class-ph': 'physics',
        'physics.comp-ph': 'physics',
        'physics.data-an': 'physics',
        'physics.ed-ph': 'physics',
        'physics.flu-dyn': 'physics',
        'physics.gen-ph': 'physics',
        'physics.geo-ph': 'physics',
        'physics.hist-ph': 'physics',
        'physics.ins-det': 'physics',
        'physics.med-ph': 'physics',
        'physics.optics': 'physics',
        'physics.plasm-ph': 'physics',
        'physics.pop-ph': 'physics',
        'physics.soc-ph': 'physics',
        'physics.space-ph': 'physics',
        'quant-ph': 'physics',
        'math': 'mathematics',
        'math.ag': 'mathematics',
        'math.ap': 'mathematics',
        'math.ac': 'mathematics',
        'math.at': 'mathematics',
        'math.ca': 'mathematics',
        'math.co': 'mathematics',
        'math.ct': 'mathematics',
        'math.cv': 'mathematics',
        'math.dg': 'mathematics',
        'math.ds': 'mathematics',
        'math.fa': 'mathematics',
        'math.gm': 'mathematics',
        'math.gn': 'mathematics',
        'math.gr': 'mathematics',
        'math.gt': 'mathematics',
        'math.ho': 'mathematics',
        'math.it': 'mathematics',
        'math.kt': 'mathematics',
        'math.lo': 'mathematics',
        'math.mg': 'mathematics',
        'math.mp': 'mathematics',
        'math.na': 'mathematics',
        'math.nt': 'mathematics',
        'math.oa': 'mathematics',
        'math.oc': 'mathematics',
        'math.pr': 'mathematics',
        'math.qa': 'mathematics',
        'math.ra': 'mathematics',
        'math.rt': 'mathematics',
        'math.sg': 'mathematics',
        'math.sp': 'mathematics',
        'math.st': 'mathematics',
        'cs': 'computer science',
        'cs.ai': 'computer science',
        'cs.ar': 'computer science',
        'cs.cc': 'computer science',
        'cs.ce': 'computer science',
        'cs.cg': 'computer science',
        'cs.cl': 'computer science',
        'cs.cr': 'computer science',
        'cs.cv': 'computer science',
        'cs.cy': 'computer science',
        'cs.db': 'computer science',
        'cs.dc': 'computer science',
        'cs.dl': 'computer science',
        'cs.dm': 'computer science',
        'cs.ds': 'computer science',
        'cs.et': 'computer science',
        'cs.fl': 'computer science',
        'cs.gl': 'computer science',
        'cs.gr': 'computer science',
        'cs.gt': 'computer science',
        'cs.hc': 'computer science',
        'cs.ir': 'computer science',
        'cs.it': 'computer science',
        'cs.lg': 'computer science',
        'cs.lo': 'computer science',
        'cs.ma': 'computer science',
        'cs.mm': 'computer science',
        'cs.ms': 'computer science',
        'cs.na': 'computer science',
        'cs.ne': 'computer science',
        'cs.ni': 'computer science',
        'cs.oh': 'computer science',
        'cs.os': 'computer science',
        'cs.pf': 'computer science',
        'cs.pl': 'computer science',
        'cs.ro': 'computer science',
        'cs.sc': 'computer science',
        'cs.sd': 'computer science',
        'cs.se': 'computer science',
        'cs.si': 'computer science',
        'cs.sy': 'computer science',
        'q-bio': 'quantitative biology',
        'q-bio.bm': 'quantitative biology',
        'q-bio.cb': 'quantitative biology',
        'q-bio.gn': 'quantitative biology',
        'q-bio.mn': 'quantitative biology',
        'q-bio.nc': 'quantitative biology',
        'q-bio.ot': 'quantitative biology',
        'q-bio.pe': 'quantitative biology',
        'q-bio.qm': 'quantitative biology',
        'q-bio.sc': 'quantitative biology',
        'q-bio.to': 'quantitative biology',
        'q-fin': 'quantitative finance',
        'q-fin.cp': 'quantitative finance',
        'q-fin.ec': 'quantitative finance',
        'q-fin.gn': 'quantitative finance',
        'q-fin.mf': 'quantitative finance',
        'q-fin.pm': 'quantitative finance',
        'q-fin.pr': 'quantitative finance',
        'q-fin.rm': 'quantitative finance',
        'q-fin.st': 'quantitative finance',
        'q-fin.tr': 'quantitative finance',
        'stat': 'statistics',
        'stat.ap': 'statistics',
        'stat.co': 'statistics',
        'stat.me': 'statistics',
        'stat.ml': 'statistics',
        'stat.ot': 'statistics',
        'stat.th': 'statistics',
        'eess': 'electrical engineering and systems science',
        'eess.as': 'electrical engineering and systems science',
        'eess.iv': 'electrical engineering and systems science',
        'eess.sp': 'electrical engineering and systems science',
        'eess.sy': 'electrical engineering and systems science',
        'econ': 'economics',
        'econ.em': 'economics',
        'econ.gn': 'economics',
        'econ.th': 'economics'
    }

def __plot_category_distribution(df):
    """Plot distribution of papers across categories."""
    if len(df) == 0:
        return
        
    category_counts = df['category'].value_counts()
    
    plt.figure(figsize=(12, 6))
    category_counts.sort_values(ascending=False).plot(kind='bar')
    plt.title('Distribution of Papers Across Categories')
    plt.xlabel('Category')
    plt.ylabel('Number of Papers')
    plt.xticks(rotation=90)
    
    for i, v in enumerate(category_counts.sort_values(ascending=False)):
        plt.text(i, v, str(v), ha='center', va='bottom')
    
    plt.ylim(0, category_counts.max() * 1.1)
    plt.tight_layout()
    clear_output(wait=True)
    plt.show()

def __create_paper_dataframe(results):
    """Create DataFrame from arxiv results."""
    return pd.DataFrame([{
        "id": result.entry_id if result.entry_id else "",
        "date": result.published if result.published else "",
        "title": __safe_decode(result.title) if result.title else "",
        "summary": __clean_summary(__safe_decode(result.summary)) if result.summary else "",
        "comment": __safe_decode(result.comment) if result.comment else "",
        "authors": ", ".join(__safe_decode(str(author)) for author in result.authors) if result.authors else "",
        "category": result.primary_category if result.primary_category else "",
    } for result in results])

def __map_categories(df, category_mapping):
    """Map categories to their parent categories. Unmapped categories default to 'physics'."""
    for index, row in df.iterrows():
        try:
            df.at[index, 'category'] = category_mapping.get(row['category'].lower(), 'physics')
        except:
            df.at[index, 'category'] = 'physics'
    return df

def __split_dataset(df):
    """Split dataset into train, validation and test sets."""
    if df['category'].isna().any():
        print("Warning: NaN values found in 'category' column. Removing rows with NaN values.")
        df = df.dropna(subset=['category'])

    train_val, test = train_test_split(df, test_size=0.1, random_state=42, stratify=df['category'])
    train, val = train_test_split(train_val, test_size=0.3, random_state=42, stratify=train_val['category'])

    test["split"] = "test"
    train["split"] = "train"
    val["split"] = "val"

    return pd.concat([train, val, test])

def data(max_results=5_000):
    """Main function to scrape and process arxiv data."""
    df = pd.DataFrame(columns=["title", "summary", "comment", "authors", "category", "split"])
    category_mapping = __get_category_mapping()
    categories = category_mapping.keys()

    for category in tqdm(categories, desc="Processing categories"):
        client = arxiv.Client()
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        results = __fetch_results(client, search)

        df_temp = __create_paper_dataframe(results)
        df_temp = __map_categories(df_temp, category_mapping)
        
        df = pd.concat([df, df_temp], ignore_index=True)
        df = df.drop_duplicates(subset=['id'])
        df = df.reset_index(drop=True)
        
        __plot_category_distribution(df)

    df = df.drop(columns=['id'])
    df = __split_dataset(df)
    df = df.sort_index().reset_index(drop=True)

    print(df['split'].value_counts(normalize=True))

    if df.isna().any().any():
        print("Warning: NaN values still present in the DataFrame")
        print(df.isna().sum())

    df.to_csv(f'../data/arxiv_{max_results}.csv', index=False, encoding='utf-8')
    return df