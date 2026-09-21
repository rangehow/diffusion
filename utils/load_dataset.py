import datasets
import os
import json
import re
from functools import wraps
from typing import Dict, Callable, Any, Optional
from pathlib import Path

# 全局数据集注册表
_DATASET_REGISTRY: Dict[str, Callable] = {}

# 数据集本地路径配置
_LOCAL_PATHS: Dict[str, str] = {}

def load_dataset_config(config_path: str = "diffusion/dataset_config.json"):
    """
    从配置文件加载数据集本地路径映射
    """
    global _LOCAL_PATHS
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                _LOCAL_PATHS = config.get('local_paths', {})
                print(f"已加载数据集配置: {len(_LOCAL_PATHS)} 个本地路径")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            _LOCAL_PATHS = {}
    else:
        print(f"配置文件 {config_path} 不存在，将使用默认远程加载")
        _LOCAL_PATHS = {}

def register_dataset(remote_loader: Optional[Callable] = None):
    """
    数据集注册装饰器
    支持本地路径优先加载
    """
    def decorator(func: Callable) -> Callable:
        dataset_name = func.__name__
        
        if dataset_name in _DATASET_REGISTRY:
            raise ValueError(f"数据集 '{dataset_name}' 已经存在，请使用不同的函数名")
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            local_path = _LOCAL_PATHS.get(dataset_name)
            return func(local_path, *args, **kwargs)
        
        _DATASET_REGISTRY[dataset_name] = wrapper
        return wrapper
    
    return decorator

def get_dataset(name: str, *args, **kwargs) -> Any:
    """
    根据名称获取数据集
    """
    if name not in _DATASET_REGISTRY:
        available_datasets = list(_DATASET_REGISTRY.keys())
        raise ValueError(f"数据集 '{name}' 未找到。可用数据集: {available_datasets}")
    
    return _DATASET_REGISTRY[name](*args, **kwargs)

def list_datasets() -> list:
    """
    列出所有已注册的数据集名称
    """
    return list(_DATASET_REGISTRY.keys())

def get_local_path(dataset_name: str) -> Optional[str]:
    """
    获取数据集的本地路径
    """
    return _LOCAL_PATHS.get(dataset_name)


# ============================================================================
# 文本后处理工具函数
# ============================================================================

def lm1b_detokenizer(x: str) -> str:
    """
    Detokenizer function - 修复 tokenized 文本中的空格问题
    """
    # 第一步：处理 URL 格式 (http : / / xxx . xxx . xxx)
    # 先处理协议部分
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    
    # 处理 URL 中域名的点 (www . example . com -> www.example.com)
    # 这个正则匹配 http(s):// 后面直到空格或句子结束的部分
    def fix_url(match):
        url_start = match.group(1)  # http:// or https://
        url_rest = match.group(2)   # 域名和路径部分
        # 移除域名部分中点前后的空格
        url_rest = re.sub(r'\s*\.\s*', '.', url_rest)
        # 移除斜杠前后的空格
        url_rest = re.sub(r'\s*/\s*', '/', url_rest)
        return url_start + url_rest
    
    x = re.sub(r'(https?://)([^\s]+(?:\s*\.\s*\w+)*)', fix_url, x)
    
    # 处理 n't 缩写形式 (was n't -> wasn't, could n't -> couldn't, etc.)
    x = re.sub(r" n't\b", "n't", x)
    
    # 处理其他缩写形式 (如 i 'll -> i'll, don 't -> don't, they 're -> they're)
    x = re.sub(r" '(\w+)", r"'\1", x)
    
    # 处理连字符 (short - term -> short-term)
    x = re.sub(r' - ', '-', x)
    
    # 处理句末标点 (保留后面的空格)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)  # 句中
    x = re.sub(r' (\w+) \.$', r' \1.', x)   # 句末
    
    # 处理问号和感叹号 (保留后面的空格)
    x = re.sub(r' \? ', '? ', x)
    x = re.sub(r' \?$', '?', x)
    x = re.sub(r' ! ', '! ', x)
    x = re.sub(r' !$', '!', x)
    
    # 处理逗号、冒号、分号
    x = x.replace(' , ', ', ')
    x = re.sub(r' : (?!//)', ': ', x)  # 排除 URL 中的冒号
    x = x.replace(' ; ', '; ')
    
    # 处理斜杠 (在 URL 之外的斜杠)
    # 简单处理：如果斜杠不是紧跟在冒号后面，移除前后空格
    def fix_slash(match):
        before = match.group(1)
        after = match.group(2)
        # 如果前面是冒号（URL的一部分），保持原样
        if before.endswith(':'):
            return match.group(0)
        return before + '/' + after
    
    x = re.sub(r'(\S) / (\S)', fix_slash, x)
    
    # 处理引号
    x = re.sub(r'" ([^"]+) "', r'"\1"', x)
    x = re.sub(r"' ([^']+) '", r"'\1'", x)
    
    # 处理括号
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    
    # 处理货币符号
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    x = x.replace('€ ', '€')
    
    # 清理多余空格
    x = re.sub(r'  +', ' ', x)
    
    return x


def simple_truecaser(text: str) -> str:
    """
    简单的 Truecasing - 恢复基本的大小写规则
    
    处理规则:
    1. 句首字母大写
    2. "I" 作为独立代词时大写
    3. 常见专有名词/缩写大写 (可扩展)
    """
    if not text:
        return text
    
    # 常见需要大写的词汇 (可根据需要扩展)
    ALWAYS_UPPER = {
        'i': 'I',  # 代词 I
        'u.s.': 'U.S.',
        'u.s.a.': 'U.S.A.',
        'uk': 'UK',
        'u.k.': 'U.K.',
        'eu': 'EU',
        'un': 'UN',
        'nato': 'NATO',
        'nasa': 'NASA',
        'fbi': 'FBI',
        'cia': 'CIA',
        'ceo': 'CEO',
        'cfo': 'CFO',
        'cto': 'CTO',
        'ai': 'AI',
        'ml': 'ML',
        'nlp': 'NLP',
        'api': 'API',
        'url': 'URL',
        'html': 'HTML',
        'css': 'CSS',
        'xml': 'XML',
        'json': 'JSON',
        'sql': 'SQL',
        'http': 'HTTP',
        'https': 'HTTPS',
        'gpu': 'GPU',
        'cpu': 'CPU',
        'ram': 'RAM',
        'rom': 'ROM',
        'usb': 'USB',
        'pdf': 'PDF',
        'ok': 'OK',
        'tv': 'TV',
        'dna': 'DNA',
        'rna': 'RNA',
        'phd': 'PhD',
        'mr.': 'Mr.',
        'mrs.': 'Mrs.',
        'ms.': 'Ms.',
        'dr.': 'Dr.',
        'prof.': 'Prof.',
    }
    
    # 常见月份和星期
    MONTHS_DAYS = {
        'january': 'January', 'february': 'February', 'march': 'March',
        'april': 'April', 'may': 'May', 'june': 'June',
        'july': 'July', 'august': 'August', 'september': 'September',
        'october': 'October', 'november': 'November', 'december': 'December',
        'monday': 'Monday', 'tuesday': 'Tuesday', 'wednesday': 'Wednesday',
        'thursday': 'Thursday', 'friday': 'Friday', 'saturday': 'Saturday',
        'sunday': 'Sunday',
        'jan': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'apr': 'Apr',
        'jun': 'Jun', 'jul': 'Jul', 'aug': 'Aug', 'sep': 'Sep',
        'oct': 'Oct', 'nov': 'Nov', 'dec': 'Dec',
        'mon': 'Mon', 'tue': 'Tue', 'wed': 'Wed', 'thu': 'Thu',
        'fri': 'Fri', 'sat': 'Sat', 'sun': 'Sun',
    }
    
    ALWAYS_UPPER.update(MONTHS_DAYS)
    
    # 常见缩写模式 (不应被视为句子结束)
    ABBREVIATIONS = {
        'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.',
        'u.s.', 'u.s.a.', 'u.k.', 'e.g.', 'i.e.', 'etc.',
        'vs.', 'inc.', 'ltd.', 'corp.', 'co.',
        'jan.', 'feb.', 'mar.', 'apr.', 'jun.', 'jul.', 
        'aug.', 'sep.', 'oct.', 'nov.', 'dec.',
        'st.', 'ave.', 'blvd.', 'rd.', 'apt.', 'no.',
    }
    
    # 不应该修改大小写的模式 (如 URL)
    URL_PATTERN = re.compile(r'https?://\S+', re.IGNORECASE)
    
    def process_word(word: str, is_sentence_start: bool) -> str:
        """处理单个词"""
        # 如果是 URL，不处理
        if URL_PATTERN.match(word):
            return word
        
        # 先检查完整的词是否在字典中（处理 u.s. 这种情况）
        word_lower_full = word.lower()
        if word_lower_full in ALWAYS_UPPER:
            return ALWAYS_UPPER[word_lower_full]
        
        # 分离词尾的标点符号（但保留缩写的点，如 u.s. 中间的点）
        trailing_punct = ''
        word_core = word
        # 只剥离非字母数字和非点的标点，或者句末的单个点
        while word_core and word_core[-1] in ',!?;:"\')]}':
            trailing_punct = word_core[-1] + trailing_punct
            word_core = word_core[:-1]
        
        # 特殊处理：如果词以点结尾，检查去掉点后是否仍然匹配缩写
        if word_core and word_core[-1] == '.':
            # 检查是否是已知缩写
            if word_core.lower() not in ALWAYS_UPPER and word_core.lower() not in ABBREVIATIONS:
                # 不是缩写，点是句号
                trailing_punct = '.' + trailing_punct
                word_core = word_core[:-1]
        
        # 分离词首的标点符号
        leading_punct = ''
        while word_core and word_core[0] in '"\'([{':
            leading_punct += word_core[0]
            word_core = word_core[1:]
        
        if not word_core:
            return word
        
        word_lower = word_core.lower()
        
        # 处理带缩写的 "I" (如 i'm, i'll, i've, i'd)
        if word_lower in ("i'm", "i'll", "i've", "i'd"):
            return leading_punct + "I" + word_core[1:] + trailing_punct
        
        # 检查是否是需要特定大写的词
        if word_lower in ALWAYS_UPPER:
            return leading_punct + ALWAYS_UPPER[word_lower] + trailing_punct
        
        # 句首大写
        if is_sentence_start and word_core and word_core[0].isalpha():
            return leading_punct + word_core[0].upper() + word_core[1:] + trailing_punct
        
        return word
    
    def is_abbreviation(word: str) -> bool:
        """检查词是否是缩写"""
        # 去除尾部的标点（除了缩写本身的点）
        word_clean = word.rstrip('.,!?;:"\')}]')
        # 如果去除后为空或者不以点结尾，再尝试加上点
        if word_clean.lower() in ABBREVIATIONS:
            return True
        if not word_clean.endswith('.'):
            word_clean += '.'
        return word_clean.lower() in ABBREVIATIONS
    
    # 先按空格分词处理
    words = text.split(' ')
    result_words = []
    is_sentence_start = True  # 文本开头视为句子开始
    
    for i, word in enumerate(words):
        if not word:
            result_words.append(word)
            continue
        
        # 处理当前词
        processed = process_word(word, is_sentence_start)
        result_words.append(processed)
        
        # 判断下一个词是否是句首
        # 如果当前词以句子结束标点结尾，且不是缩写
        word_stripped = word.rstrip()
        if word_stripped and word_stripped[-1] in '.!?' and not is_abbreviation(word_stripped):
            is_sentence_start = True
        else:
            is_sentence_start = False
    
    return ' '.join(result_words)


def advanced_truecaser(text: str) -> str:
    """
    更高级的 Truecasing - 使用额外的启发式规则
    
    额外规则:
    1. 引号后的句首大写
    2. 冒号后可能的大写 (如 "Note: This is...")
    3. 括号内句首
    """
    if not text:
        return text
    
    # 先应用简单 truecaser
    text = simple_truecaser(text)
    
    # 处理引号后的句首
    # 例如: he said, "this is important" -> he said, "This is important"
    def capitalize_after_quote(match):
        quote = match.group(1)
        char = match.group(2)
        return quote + char.upper()
    
    text = re.sub(r'(["\'])\s*([a-z])', capitalize_after_quote, text)
    
    # 处理冒号后的大写 (当后面是完整句子时)
    def capitalize_after_colon(match):
        colon_space = match.group(1)
        char = match.group(2)
        return colon_space + char.upper()
    
    text = re.sub(r'(:\s+)([a-z])', capitalize_after_colon, text)
    
    return text


def normalize_text(text: str, 
                   detokenize: bool = True, 
                   truecase: bool = True,
                   truecase_mode: str = 'simple') -> str:
    """
    统一的文本规范化函数
    
    Args:
        text: 输入文本
        detokenize: 是否应用 detokenization (修复空格)
        truecase: 是否应用 truecasing (恢复大小写)
        truecase_mode: 'simple' 或 'advanced'
    
    Returns:
        处理后的文本
    """
    if not text:
        return text
    
    if detokenize:
        text = lm1b_detokenizer(text)
    
    if truecase:
        if truecase_mode == 'advanced':
            text = advanced_truecaser(text)
        else:
            text = simple_truecaser(text)
    
    return text


def create_text_normalizer_batch(detokenize: bool = True, 
                                  truecase: bool = True,
                                  truecase_mode: str = 'simple',
                                  text_column: str = 'text'):
    """
    创建批处理版本的文本规范化函数，用于 dataset.map()
    
    Args:
        detokenize: 是否应用 detokenization
        truecase: 是否应用 truecasing
        truecase_mode: 'simple' 或 'advanced'
        text_column: 文本列名
    
    Returns:
        可用于 dataset.map() 的批处理函数
    """
    def batch_normalizer(examples):
        examples[text_column] = [
            normalize_text(t, detokenize=detokenize, truecase=truecase, truecase_mode=truecase_mode)
            for t in examples[text_column]
        ]
        return examples
    
    return batch_normalizer


def apply_text_normalization(dataset, 
                              detokenize: bool = True,
                              truecase: bool = True,
                              truecase_mode: str = 'simple',
                              text_column: str = 'text',
                              num_proc: int = 64,
                              desc: str = "Normalizing text"):
    """
    对数据集应用文本规范化
    
    Args:
        dataset: Hugging Face 数据集
        detokenize: 是否应用 detokenization
        truecase: 是否应用 truecasing
        truecase_mode: 'simple' 或 'advanced'
        text_column: 文本列名
        num_proc: 并行处理进程数
        desc: 进度条描述
    
    Returns:
        处理后的数据集
    """
    normalizer = create_text_normalizer_batch(
        detokenize=detokenize,
        truecase=truecase,
        truecase_mode=truecase_mode,
        text_column=text_column
    )
    
    return dataset.map(
        normalizer,
        batched=True,
        num_proc=num_proc,
        desc=desc
    )


# ============================================================================
# 辅助函数：检测文本是否需要规范化
# ============================================================================

def detect_tokenization_issues(text: str) -> dict:
    """
    检测文本是否存在 tokenization 问题
    
    Returns:
        dict with keys:
        - needs_detokenization: bool
        - needs_truecasing: bool
        - issues: list of detected issues
    """
    issues = []
    needs_detokenization = False
    needs_truecasing = False
    
    # 检查标点符号前的空格 (tokenization 问题的标志)
    if re.search(r' [.,!?;:]', text):
        issues.append("空格在标点符号前")
        needs_detokenization = True
    
    # 检查 http : / / 格式
    if 'http : / /' in text or 'https : / /' in text:
        issues.append("URL被tokenize分隔")
        needs_detokenization = True
    
    # 检查引号空格问题
    if re.search(r'" [^"]+ "', text) or re.search(r"' [^']+ '", text):
        issues.append("引号内有额外空格")
        needs_detokenization = True
    
    # 检查是否全小写 (但排除全大写和正常混合的情况)
    if text and text == text.lower() and any(c.isalpha() for c in text):
        # 进一步检查：是否句首应该大写但没有
        sentences = re.split(r'[.!?]\s+', text)
        if sentences and sentences[0] and sentences[0][0].islower():
            issues.append("可能全小写，需要truecasing")
            needs_truecasing = True
    
    return {
        'needs_detokenization': needs_detokenization,
        'needs_truecasing': needs_truecasing,
        'issues': issues
    }


def auto_detect_and_normalize(dataset, 
                               text_column: str = 'text',
                               sample_size: int = 100,
                               num_proc: int = 64):
    """
    自动检测数据集是否需要规范化，并应用必要的处理
    
    Args:
        dataset: Hugging Face 数据集
        text_column: 文本列名
        sample_size: 用于检测的样本数量
        num_proc: 并行处理进程数
    
    Returns:
        处理后的数据集
    """
    # 采样检测
    sample_texts = dataset.select(range(min(sample_size, len(dataset))))[text_column]
    
    detok_count = 0
    truecase_count = 0
    
    for text in sample_texts:
        if text:
            result = detect_tokenization_issues(text)
            if result['needs_detokenization']:
                detok_count += 1
            if result['needs_truecasing']:
                truecase_count += 1
    
    needs_detokenization = detok_count > sample_size * 0.1  # 超过10%的样本需要
    needs_truecasing = truecase_count > sample_size * 0.1
    
    print(f"自动检测结果: detokenization={needs_detokenization} ({detok_count}/{sample_size}), "
          f"truecasing={needs_truecasing} ({truecase_count}/{sample_size})")
    
    if needs_detokenization or needs_truecasing:
        return apply_text_normalization(
            dataset,
            detokenize=needs_detokenization,
            truecase=needs_truecasing,
            text_column=text_column,
            num_proc=num_proc
        )
    
    return dataset


# 初始化时加载配置
load_dataset_config()


# ============================================================================
# 数据集注册 - 原有数据集
# ============================================================================

@register_dataset()
def ultra_fineweb(local_path):
    """加载 Ultra-FineWeb 数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'ultra_fineweb': {local_path}")
        dataset = datasets.load_dataset(local_path, name='default', split='en[:10%]', num_proc=64)
    else:
        print("从远程加载数据集 'ultra_fineweb'")
        dataset = datasets.load_dataset("openbmb/Ultra-FineWeb", 'en', num_proc=64)
    
    dataset = dataset.rename_column('content', 'text')
    return dataset


@register_dataset()
def finefineweb(local_path):
    """加载 FineFineWeb 训练数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'finefineweb': {local_path}")
        dataset = datasets.load_dataset(local_path, num_proc=64)['train']
    else:
        print("从远程加载数据集 'finefineweb'")
        dataset = datasets.load_dataset("m-a-p/FineFineWeb-sample", num_proc=64)['train']
    
    return dataset


@register_dataset()
def finefineweb_validation(local_path):
    """加载 FineFineWeb 验证数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'finefineweb_validation': {local_path}")
        dataset = datasets.load_dataset(local_path, num_proc=64, split='train[:50%]')
    else:
        print("从远程加载数据集 'finefineweb_validation'")
        dataset = datasets.load_dataset("m-a-p/FineFineWeb-validation", num_proc=64, split='train[:50%]')
    
    return dataset


@register_dataset()
def filtered_finefineweb(local_path):
    """加载 filtered_finefineweb 数据集"""
    print(f"从本地路径加载数据集 'filtered_finefineweb': {local_path}")
    dataset = datasets.load_from_disk(local_path)
    return dataset


@register_dataset()
def debug(local_path):
    dataset = datasets.Dataset.from_dict({
        "text": ["Today is a really good day isn't it? Wanna go for a walk?", "hello world"]
    })
    return dataset


@register_dataset()
def fineweb_edu_10b(local_path):
    """加载 fineweb-edu-dedup-10b 数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'fineweb_edu_10b': {local_path}")
        try:
            return datasets.load_dataset(local_path, num_proc=64, verification_mode='no_checks')['train']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
    
    print("从远程加载数据集 'fineweb_edu_10b'")
    return datasets.load_dataset("EleutherAI/fineweb-edu-dedup-10b")['train']


@register_dataset()
def fineweb_edu_1b(local_path):
    """加载 fineweb-edu-1B 数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'fineweb_edu_1b': {local_path}")
        try:
            return datasets.load_dataset(local_path, num_proc=64, verification_mode='no_checks')['train']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
    
    print("从远程加载数据集 'fineweb_edu_1b'")
    return datasets.load_dataset("codelion/fineweb-edu-1B")['train']


@register_dataset()
def common_crawl(local_path):
    """加载 Common Crawl 数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'common_crawl': {local_path}")
        try:
            return datasets.load_from_disk(local_path)
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
    
    print("从远程加载数据集 'common_crawl'")
    return datasets.load_dataset("common_crawl")


@register_dataset()
def wikipedia(local_path):
    """加载 Wikipedia 数据集"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'wikipedia': {local_path}")
        try:
            return datasets.load_from_disk(local_path)
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
    
    return datasets.load_dataset("wikipedia", "20220301.en")


@register_dataset()
def smoltalk2_midtrain(local_path):
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'smoltalk2_midtrain': {local_path}")
        try:
            return datasets.load_dataset(local_path, num_proc=64, verification_mode='no_checks')['train']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
    
    return datasets.load_dataset(local_path, num_proc=64, verification_mode='no_checks')['train']


# ============================================================================
# 需要特殊处理的评测数据集 (MMLU, GSM8K, etc.)
# ============================================================================

@register_dataset()
def mmlu(local_path=None):
    """加载 MMLU 数据集并处理成 text 列"""
    def _format_example(example: dict) -> str:
        choices_letters = ["A", "B", "C", "D"]
        prompt = "Question: " + example["question"]
        for i, choice_text in enumerate(example["choices"]):
            prompt += f"\n{choices_letters[i]}. {choice_text}"
        answer_index = int(example["answer"])
        correct_letter = choices_letters[answer_index]
        prompt += "\nAnswer: " + correct_letter
        return prompt

    def preprocess(example):
        formatted_qa = _format_example(example['train'])
        return {"text": formatted_qa}

    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'mmlu': {local_path}")
        ds = datasets.load_dataset(local_path, 'auxiliary_train')['train']
    else:
        print("从 Hugging Face Hub 远程加载数据集 mmlu")
        ds = datasets.load_dataset('cais/mmlu', 'auxiliary_train', split='train', num_proc=64)

    processed_ds = ds.map(
        preprocess,
        num_proc=64,
        remove_columns=ds.column_names,
        desc="Formatting MMLU dataset for training"
    )
    return processed_ds


@register_dataset()
def gsm8k(local_path=None):
    """加载 GSM8K 数据集并处理成 text 列"""
    def preprocess(example):
        text = f"Question: {example['question']}\nAnswer: {example['answer']}"
        return {"text": text}

    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'gsm8k': {local_path}")
        try:
            ds = datasets.load_dataset(local_path, 'main', num_proc=64)['train']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
            ds = datasets.load_dataset('openai/gsm8k', 'main', num_proc=64)['train']
    else:
        print("从 Hugging Face Hub 远程加载数据集 'gsm8k'")
        ds = datasets.load_dataset('openai/gsm8k', 'main', num_proc=64)['train']

    processed_ds = ds.map(preprocess, num_proc=64, remove_columns=ds.column_names)
    return processed_ds


@register_dataset()
def winogrande(local_path=None):
    """加载 Winogrande 数据集并处理成 text 列"""
    def preprocess(example):
        correct_option = example['option1'] if example['answer'] == '1' else example['option2']
        text = example['sentence'].replace('_', correct_option)
        return {"text": text}

    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'winogrande': {local_path}")
        try:
            ds = datasets.load_dataset(local_path, 'winogrande_debiased', num_proc=64)['train']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
            ds = datasets.load_dataset('allenai/winogrande', 'winogrande_debiased', num_proc=64)['train']
    else:
        print("从 Hugging Face Hub 远程加载数据集 'winogrande'")
        ds = datasets.load_dataset('allenai/winogrande', 'winogrande_debiased', num_proc=64)['train']
        
    processed_ds = ds.map(preprocess, num_proc=64, remove_columns=ds.column_names)
    return processed_ds


@register_dataset()
def python(local_path):
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'python': {local_path}")
        try:
            return datasets.load_dataset(local_path, num_proc=64)['train'].shuffle().take(10000)
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
    
    return datasets.load_dataset('Avelina/python-edu-cleaned', num_proc=64)['train'].shuffle().take(10000)


@register_dataset()
def hellaswag(local_path=None):
    """加载 Hellaswag 数据集"""
    def preprocess(example):
        context = example['ctx']
        correct_ending = example['endings'][int(example['label'])]
        text = f"{context} {correct_ending}"
        return {"text": text}

    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'hellaswag': {local_path}")
        try:
            ds = datasets.load_dataset(local_path, num_proc=64)['train']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
            ds = datasets.load_dataset('Rowan/hellaswag', num_proc=64)['train']
    else:
        print("从 Hugging Face Hub 远程加载数据集 'hellaswag'")
        ds = datasets.load_dataset('Rowan/hellaswag', num_proc=64)['train']

    processed_ds = ds.map(preprocess, num_proc=64, remove_columns=ds.column_names)
    return processed_ds


@register_dataset()
def dolmino(local_path):
    subset_name = ['stackexchange', 'math', 'wiki']
    dataset_list = []

    for subset in subset_name:
        if local_path is not None and os.path.exists(local_path):
            print(f"从本地路径加载数据集 'dolmino/{subset}': {local_path}")
            try:
                dataset = datasets.load_dataset(local_path, subset, num_proc=64)['train'].shuffle().take(10000)
            except Exception as e:
                print(f"本地加载失败: {e}, 尝试远程加载")
                dataset = datasets.load_dataset('allenai/dolmino-mix-1124', subset, num_proc=64)['train'].shuffle().take(10000)
        else:
            dataset = datasets.load_dataset('allenai/dolmino-mix-1124', subset, num_proc=64)['train'].shuffle().take(10000)
        dataset_list.append(dataset)

    return datasets.concatenate_datasets(dataset_list)


# ============================================================================
# LM1B 数据集 (需要 detokenization)
# ============================================================================

def _apply_lm1b_detokenizer_to_batch(examples):
    """批处理版本的 LM1B detokenizer"""
    examples['text'] = [lm1b_detokenizer(t) for t in examples['text']]
    return examples


@register_dataset()
def lm1b_train(local_path):
    """加载 LM1B 训练数据集"""
    if local_path is not None and os.path.exists(local_path):
        dataset = datasets.load_dataset(local_path, num_proc=64)['train']
    else:
        dataset = datasets.load_dataset("dvruette/lm1b")['train']
    
    dataset = dataset.map(
        _apply_lm1b_detokenizer_to_batch,
        batched=True,
        desc="Applying LM1B detokenizer to train set"
    )
    return dataset


@register_dataset()
def lm1b_test(local_path):
    """加载 LM1B 测试数据集"""
    if local_path is not None and os.path.exists(local_path):
        dataset = datasets.load_dataset(local_path, num_proc=64)['test']
    else:
        dataset = datasets.load_dataset("dvruette/lm1b")['test']

    dataset = dataset.map(
        _apply_lm1b_detokenizer_to_batch,
        batched=True,
        desc="Applying LM1B detokenizer to test set"
    )
    return dataset


# ============================================================================
# sh_ 前缀数据集 - 统一添加文本规范化处理
# ============================================================================

def _create_sh_normalizer_batch(text_column: str = 'text'):
    """
    创建 sh_ 数据集专用的规范化函数
    同时应用 detokenization 和 truecasing
    """
    def normalizer(examples):
        if text_column in examples:
            examples[text_column] = [
                normalize_text(t, detokenize=True, truecase=True, truecase_mode='simple')
                if t else t
                for t in examples[text_column]
            ]
        return examples
    return normalizer


@register_dataset()
def sh_filtered_finefineweb(local_path):
    """加载 sh_filtered_finefineweb 数据集"""
    print(f"从本地路径加载数据集 'sh_filtered_finefineweb': {local_path}")
    dataset = datasets.load_from_disk(local_path)
    return dataset


@register_dataset()
def sh_fineweb_edu_10b(local_path):
    """加载 sh_fineweb_edu_10b 数据集 (带文本规范化)"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'sh_fineweb_edu_10b': {local_path}")
        try:
            dataset = datasets.load_dataset(local_path, num_proc=64, verification_mode='no_checks')['train']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
            dataset = datasets.load_dataset("EleutherAI/fineweb-edu-dedup-10b")['train']
    else:
        print("从远程加载数据集 'sh_fineweb_edu_10b'")
        dataset = datasets.load_dataset("EleutherAI/fineweb-edu-dedup-10b")['train']
    
    return dataset


@register_dataset()
def sh_fineweb_edu_1b(local_path):
    """加载 sh_fineweb_edu_1b 数据集 (带文本规范化)"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'sh_fineweb_edu_1b': {local_path}")
        try:
            dataset = datasets.load_dataset(local_path, num_proc=64, verification_mode='no_checks')['train']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
            dataset = datasets.load_dataset("codelion/fineweb-edu-1B")['train']
    else:
        print("从远程加载数据集 'sh_fineweb_edu_1b'")
        dataset = datasets.load_dataset("codelion/fineweb-edu-1B")['train']
    
    return dataset


@register_dataset()
def sh_lm1b_test(local_path):
    """加载 sh_lm1b_test 数据集 (带文本规范化)"""
    if local_path is not None and os.path.exists(local_path):
        dataset = datasets.load_dataset(local_path, num_proc=64)['test']
    else:
        dataset = datasets.load_dataset("dvruette/lm1b")['test']

    return dataset


@register_dataset()
def sh_openwebtext(local_path):
    """加载 sh_openwebtext 数据集 (带文本规范化)"""
    if local_path is not None and os.path.exists(local_path):
        try:
            dataset = datasets.load_dataset(local_path, num_proc=64)['train']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
            dataset = datasets.load_dataset("Skylion007/openwebtext")['train']
    else:
        dataset = datasets.load_dataset("Skylion007/openwebtext")['train']

    dataset = dataset.select(range(len(dataset)-100000, len(dataset)))
    
    return dataset


@register_dataset()
def sh_lambada(local_path):
    """加载 sh_lambada 测试数据集 (带文本规范化)"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'sh_lambada': {local_path}")
        try:
            dataset = datasets.load_dataset(local_path, num_proc=64)['test']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
            dataset = datasets.load_dataset("EleutherAI/lambada_openai", num_proc=64)['test']
    else:
        print("从远程加载数据集 'sh_lambada'")
        dataset = datasets.load_dataset("EleutherAI/lambada_openai", num_proc=64)['test']
    
    return dataset


@register_dataset()
def sh_ptb(local_path):
    """加载 sh_ptb (Penn Treebank) 测试数据集 (带文本规范化)"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'sh_ptb': {local_path}")
        try:
            dataset = datasets.load_dataset(local_path, num_proc=64)['test']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
            dataset = datasets.load_dataset("ptb-text-only/ptb_text_only", num_proc=64)['test']
    else:
        print("从远程加载数据集 'sh_ptb'")
        dataset = datasets.load_dataset("ptb-text-only/ptb_text_only", num_proc=64)['test']
    
    # Rename 'sentence' column to 'text' for consistency
    text_col = 'sentence' if 'sentence' in dataset.column_names else 'text'
    if text_col == 'sentence':
        dataset = dataset.rename_column('sentence', 'text')
    
    return dataset


@register_dataset()
def sh_ag_news(local_path):
    """加载 sh_ag_news 测试数据集 (带文本规范化)"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'sh_ag_news': {local_path}")
        try:
            dataset = datasets.load_dataset(local_path, num_proc=64)['test']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
            dataset = datasets.load_dataset("fancyzhx/ag_news", num_proc=64)['test']
    else:
        print("从远程加载数据集 'sh_ag_news'")
        dataset = datasets.load_dataset("fancyzhx/ag_news", num_proc=64)['test']
    
    # 只保留 text 列
    if 'label' in dataset.column_names:
        dataset = dataset.remove_columns(['label'])
    
    return dataset


@register_dataset()
def sh_pubmed(local_path):
    """加载 sh_pubmed 测试数据集 (带文本规范化)"""
    def preprocess(example):
        text = example.get('article', '') or ''
        return {"text": text}
    
    dataset = None
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'sh_pubmed': {local_path}")
        pubmed_path = os.path.join(local_path, 'pubmed')
        if os.path.exists(pubmed_path):
            try:
                dataset = datasets.load_dataset(pubmed_path, num_proc=64)['test']
            except Exception as e:
                print(f"从子目录加载失败: {e}")
        
        if dataset is None:
            try:
                dataset = datasets.load_dataset(local_path, 'pubmed', num_proc=64)['test']
            except Exception:
                try:
                    dataset = datasets.load_dataset(local_path, num_proc=64)['test']
                except Exception as e:
                    print(f"本地加载失败: {e}, 尝试远程加载")
    
    if dataset is None:
        print("从远程加载数据集 'sh_pubmed'")
        try:
            dataset = datasets.load_dataset("armanc/scientific_papers", 'pubmed', num_proc=64, trust_remote_code=True)['test']
        except Exception:
            print("尝试备用数据集 'ccdv/pubmed-summarization'")
            dataset = datasets.load_dataset("ccdv/pubmed-summarization", num_proc=64)['test']
    
    # 处理成统一的 text 列
    if 'text' not in dataset.column_names:
        dataset = dataset.map(preprocess, num_proc=64, remove_columns=dataset.column_names)
    
    return dataset


@register_dataset()
def sh_arxiv(local_path):
    """加载 sh_arxiv 测试数据集 (带文本规范化)"""
    def preprocess(example):
        text = example.get('article', '') or ''
        return {"text": text}
    
    dataset = None
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'sh_arxiv': {local_path}")
        arxiv_path = os.path.join(local_path, 'arxiv')
        if os.path.exists(arxiv_path):
            try:
                dataset = datasets.load_dataset(arxiv_path, num_proc=64)['test']
            except Exception as e:
                print(f"从子目录加载失败: {e}")
        
        if dataset is None:
            try:
                dataset = datasets.load_dataset(local_path, 'arxiv', num_proc=64)['test']
            except Exception:
                try:
                    dataset = datasets.load_dataset(local_path, num_proc=64)['test']
                except Exception as e:
                    print(f"本地加载失败: {e}, 尝试远程加载")
    
    if dataset is None:
        print("从远程加载数据集 'sh_arxiv'")
        try:
            dataset = datasets.load_dataset("armanc/scientific_papers", 'arxiv', num_proc=64, trust_remote_code=True)['test']
        except Exception:
            print("尝试备用数据集 'ccdv/arxiv-summarization'")
            dataset = datasets.load_dataset("ccdv/arxiv-summarization", num_proc=64)['test']
    
    # 处理成统一的 text 列
    if 'text' not in dataset.column_names:
        dataset = dataset.map(preprocess, num_proc=64, remove_columns=dataset.column_names)
    
    return dataset


@register_dataset()
def sh_wikitext(local_path):
    """加载 sh_wikitext (WikiText-103) 测试数据集 (带文本规范化)"""
    if local_path is not None and os.path.exists(local_path):
        print(f"从本地路径加载数据集 'sh_wikitext': {local_path}")
        try:
            dataset = datasets.load_dataset(local_path, 'wikitext-103-raw-v1', num_proc=64)['test']
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
            dataset = datasets.load_dataset("Salesforce/wikitext", 'wikitext-103-raw-v1', num_proc=64)['test']
    else:
        print("从远程加载数据集 'sh_wikitext'")
        dataset = datasets.load_dataset("Salesforce/wikitext", 'wikitext-103-raw-v1', num_proc=64)['test']
    
    return dataset


# ============================================================================
# OpenWebText 数据集
# ============================================================================

@register_dataset()
def owt_train(local_path):
    if local_path is not None and os.path.exists(local_path):
        try:
            dataset = datasets.load_dataset(local_path, num_proc=64)
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
            dataset = datasets.load_dataset("Skylion007/openwebtext")
    else:
        dataset = datasets.load_dataset("Skylion007/openwebtext")

    dataset = dataset.select(range(len(dataset)-100000))
    return dataset


@register_dataset()
def owt_test(local_path):
    if local_path is not None and os.path.exists(local_path):
        try:
            dataset = datasets.load_dataset(local_path, num_proc=64)
        except Exception as e:
            print(f"本地加载失败: {e}, 尝试远程加载")
            dataset = datasets.load_dataset("Skylion007/openwebtext")
    else:
        dataset = datasets.load_dataset("Skylion007/openwebtext")

    dataset = dataset.select(range(len(dataset)-100000, len(dataset)))
    return dataset


# ============================================================================
# Qwen3-8B CPT 数据集
# ============================================================================

_NEMOTRON_MATH_BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/nvidia/Nemotron-CC-Math-v1/main"
_STACK_EDU_PYTHON_BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/hongliu9903/stack_edu_python/main"
_DCLM_BASELINE_BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/kothasuhas/dclm-baseline-1.0_subset_30M/main"
_FINEWEB_EDU_MINI_BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/BERT_TRAINING_SERVICE/platform/dataset/deatos/fineweb-edu-mini-combined/main"


@register_dataset()
def nemotron_cc_math(local_path):
    """Nemotron-CC-Math 4plus subset (highest quality math/reasoning)."""
    import glob
    data_dir = os.path.join(_NEMOTRON_MATH_BASE, "4plus")
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    assert len(files) > 0, f"No parquet files found in {data_dir}"
    print(f"Loading nemotron_cc_math: {len(files)} parquet files from {data_dir}")
    dataset = datasets.load_dataset("parquet", data_files=files, split="train", num_proc=64)
    # Keep only text column
    cols_to_remove = [c for c in dataset.column_names if c != "text"]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
    return dataset


@register_dataset()
def stack_edu_python(local_path):
    """stack_edu_python — Python code dataset. Renames 'content' -> 'text'."""
    import glob
    data_dir = os.path.join(_STACK_EDU_PYTHON_BASE, "data")
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    assert len(files) > 0, f"No parquet files found in {data_dir}"
    print(f"Loading stack_edu_python: {len(files)} parquet files from {data_dir}")
    dataset = datasets.load_dataset("parquet", data_files=files, split="train", num_proc=64)
    # Rename content -> text
    dataset = dataset.rename_column("content", "text")
    # Keep only text column
    cols_to_remove = [c for c in dataset.column_names if c != "text"]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
    return dataset


@register_dataset()
def dclm_baseline_30m(local_path):
    """dclm-baseline-1.0 30M subset — large general web corpus."""
    import glob
    data_dir = os.path.join(_DCLM_BASELINE_BASE, "data")
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    assert len(files) > 0, f"No parquet files found in {data_dir}"
    print(f"Loading dclm_baseline_30m: {len(files)} parquet files from {data_dir}")
    dataset = datasets.load_dataset("parquet", data_files=files, split="train", num_proc=64)
    # Keep only text column
    cols_to_remove = [c for c in dataset.column_names if c != "text"]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
    return dataset


@register_dataset()
def fineweb_edu_mini(local_path):
    """fineweb-edu-mini-combined — education-focused web text."""
    data_file = os.path.join(_FINEWEB_EDU_MINI_BASE, "train", "train.parquet")
    assert os.path.exists(data_file), f"Parquet file not found: {data_file}"
    print(f"Loading fineweb_edu_mini: {data_file}")
    dataset = datasets.load_dataset("parquet", data_files=data_file, split="train", num_proc=64)
    # Keep only text column
    cols_to_remove = [c for c in dataset.column_names if c != "text"]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
    return dataset


@register_dataset()
def qwen3_cpt_mix(local_path):
    """
    Interleaved mixture of all 4 CPT datasets for Qwen3-8B continued pretraining.
    Mixture ratio: 20% math, 15% code, 50% web, 15% edu.
    Uses `datasets.interleave_datasets` with probability-based sampling.
    """
    print("Loading qwen3_cpt_mix — interleaving 4 datasets ...")

    ds_math = get_dataset("nemotron_cc_math")
    ds_code = get_dataset("stack_edu_python")
    ds_web  = get_dataset("dclm_baseline_30m")
    ds_edu  = get_dataset("fineweb_edu_mini")

    print(f"  nemotron_cc_math : {len(ds_math):>12,} rows")
    print(f"  stack_edu_python : {len(ds_code):>12,} rows")
    print(f"  dclm_baseline_30m: {len(ds_web):>12,} rows")
    print(f"  fineweb_edu_mini : {len(ds_edu):>12,} rows")

    mixed = datasets.interleave_datasets(
        [ds_math, ds_code, ds_web, ds_edu],
        probabilities=[0.20, 0.15, 0.50, 0.15],
        seed=42,
        stopping_strategy="all_exhausted",
    )
    print(f"  qwen3_cpt_mix    : {len(mixed):>12,} rows (interleaved)")
    return mixed


# ============================================================================
# 测试文本规范化功能
# ============================================================================
# Reasoning Task Datasets (Countdown, Sudoku)
# ============================================================================

_REASONING_DATA_BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/ruanjunhao04/diffusion/reasoning_tasks/data"


def _load_reasoning_dataset(name, split):
    """Helper to load a reasoning task dataset from disk."""
    path = os.path.join(_REASONING_DATA_BASE, f"{name}_{split}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Reasoning dataset not found at {path}. "
            f"Run `python -m reasoning_tasks.generate_data` first."
        )
    return datasets.load_from_disk(path)


@register_dataset()
def cd3_train(local_path):
    """Countdown 3-number training set"""
    return _load_reasoning_dataset("cd3", "train")

@register_dataset()
def cd3_test(local_path):
    """Countdown 3-number test set"""
    return _load_reasoning_dataset("cd3", "test")

@register_dataset()
def cd4_train(local_path):
    """Countdown 4-number training set"""
    return _load_reasoning_dataset("cd4", "train")

@register_dataset()
def cd4_test(local_path):
    """Countdown 4-number test set"""
    return _load_reasoning_dataset("cd4", "test")

@register_dataset()
def cd5_train(local_path):
    """Countdown 5-number training set"""
    return _load_reasoning_dataset("cd5", "train")

@register_dataset()
def cd5_test(local_path):
    """Countdown 5-number test set"""
    return _load_reasoning_dataset("cd5", "test")

@register_dataset()
def sudoku_train(local_path):
    """Sudoku training set"""
    return _load_reasoning_dataset("sudoku", "train")

@register_dataset()
def sudoku_test(local_path):
    """Sudoku test set"""
    return _load_reasoning_dataset("sudoku", "test")


# ============================================================================

if __name__ == "__main__":
    # 测试 detokenizer
    test_texts = [
        "this is a test . does it work ?",
        "hello , world ! how are you ?",
        "visit http : / / example.com for more info .",
        "visit http : / / www.example.com / path for more info .",
        "he said , \" this is great \" and left .",
        "the price is $ 100 .",
        "i think i 'll go home now .",
        "they 're coming and i 'm leaving .",
        "the u.s. economy is growing .",
        "no it was n't black monday .",  # 测试 n't
        "i could n't believe it !",  # 测试 n't
        "she would n't do that .",  # 测试 n't
        "for about 20 years the problem of properties of short - term changes of solar activity .",  # 测试连字符
        "this is a well - known fact .",  # 测试连字符
        "the long - term effects are unclear .",  # 测试连字符
    ]
    
    print("=" * 60)
    print("测试 Detokenization + Truecasing")
    print("=" * 60)
    
    for text in test_texts:
        normalized = normalize_text(text, detokenize=True, truecase=True)
        print(f"\n原始: {text}")
        print(f"处理后: {normalized}")
    
    print("\n" + "=" * 60)
    print("测试检测功能")
    print("=" * 60)
    
    for text in test_texts[:3]:
        result = detect_tokenization_issues(text)
        print(f"\n文本: {text[:50]}...")
        print(f"检测结果: {result}")