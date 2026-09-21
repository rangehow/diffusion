from loguru import logger
import datetime


modelType2Template = {}
think_model_set = {'qwen3'}

def register_template(cls):

    for name in cls.model_type:
        # 创建一个包装函数，自动传递 current_model_type
        def create_template_wrapper(model_type=name):
            def wrapper(tokenizer, **kwargs):
                # 如果没有显式传递 current_model_type，则使用注册时的 model_type
                if 'current_model_type' not in kwargs:
                    kwargs['current_model_type'] = model_type
                return cls(tokenizer, **kwargs)
            return wrapper
        
        modelType2Template[name] = create_template_wrapper()

    return cls  # 修复：返回 cls


class Template:
    def __init__(
        self,
        tokenizer,
        current_model_type=None,  # 新增：当前具体的模型类型
        user_token=None,
        assistant_non_loss_token=None,
        assistant_token=None,
        start_token_id=None,
        end_token_id=None,
        system_token=None,
        think_token = None,
        tool_token=None,
        efficient_eos=False,
        default_system=None,
        jinja_template=None,
        embedding_size=32768,
        merge_system_and_first_user=False, # only used by gemma series
        generation_config = None,

    ) -> None:
        self.tokenizer = tokenizer
        # 修改：简化 current_model_type 的处理逻辑，因为现在会自动传递
        self.current_model_type = current_model_type or (
            self.model_type[0] if isinstance(self.model_type, list) else self.model_type
        )
        
        self.user_token = user_token if user_token else []
        self.assistant_non_loss_token = assistant_non_loss_token if assistant_non_loss_token else []
        self.assistant_token = assistant_token if assistant_token else []
        self.system_token = system_token if system_token else []
        self.think_token = think_token if think_token else []
        self.tool_token = tool_token if tool_token else []

        self.efficient_eos = efficient_eos
        self.start_token_id = start_token_id if start_token_id else None
        self.end_token_id = end_token_id if end_token_id else None
        self.default_system = default_system if default_system else None
        self.base_eos_token_id = tokenizer.eos_token_id
        self.chat_eos_token_id = tokenizer.eos_token_id
        
        self.jinja_template = jinja_template
        self.embedding_size = embedding_size

        self.merge_system_and_first_user=merge_system_and_first_user

        self.generation_config = {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
        } if generation_config is None else generation_config

    def get_generation_config(self) -> dict:

        return self.generation_config

    def get_chat_template(self):
        return self.jinja_template

    def apply(self, messages: list[dict[str, str]],add_generation_prompt=False,enable_thinking = False):

        if self.start_token_id:
            input_id = [self.start_token_id]
            label = [-100]
        else:
            input_id, label = [], []

        start_idx = 0
        first_user_flag_for_efficient_eos = True

        

        if messages[0]["role"] == "system":
            if self.merge_system_and_first_user:
                messages[0]={"role":"user","content":messages[0]['content']+'\n\n'+messages[1]['content']}
                del messages[1]
            else:
                if self.current_model_type == "llama3.2":  # 使用 current_model_type
                    
                    today = datetime.date.today()
                    formatted_date = today.strftime("%d %b %Y")
                    content=f"""Cutting Knowledge Date: December 2023\nToday Date: {formatted_date}\n\n{messages[0]['content']}"""
                    mock_system_token="<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
                    system_token = self.tokenizer.encode(
                        mock_system_token.format_map({"content": content}),
                        add_special_tokens=False,
                    )
                
                # llama3.1的jinja可能有问题导致其不能正确获取当前时间？
                elif self.current_model_type=='llama3.1':  # 使用 current_model_type
                    content=f"""Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n{messages[0]['content']}"""
                    mock_system_token="<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
                    system_token = self.tokenizer.encode(
                        mock_system_token.format_map({"content": content}),
                        add_special_tokens=False,
                    )
                else:
                    content=messages[0]['content']
                    system_token = self.tokenizer.encode(
                        self.system_token.format_map({"content": content}),
                        add_special_tokens=False,
                    )
                input_id += system_token
                label += [-100] * len(system_token)

        elif self.default_system:
            if self.current_model_type == "llama3.1" and self.tool_token != []:  # 使用 current_model_type
                system_token = self.tokenizer.encode(
                    self.system_token.format_map(
                        {"content": "Environment: ipython\n" + self.default_system}
                    ),
                    add_special_tokens=False,
                )
            else:

                system_token = self.tokenizer.encode(
                    self.system_token.format_map({"content": self.default_system}),
                    add_special_tokens=False,
                )
            input_id += system_token
            label += [-100] * len(system_token)
        
        

        for i in range(start_idx, len(messages)):

            if messages[i]["role"] == "user":
                
                user_token = self.tokenizer.encode(
                    self.user_token.format_map({"content": messages[i]["content"]}),
                    add_special_tokens=False,
                )
                
                input_id += user_token
                if self.efficient_eos and not first_user_flag_for_efficient_eos:
                    label += [self.tokenizer.eos_token_id] + [-100] * (
                        len(user_token) - 1
                    )
                else:
                    first_user_flag_for_efficient_eos = False
                    label += [-100] * len(user_token)
            elif messages[i]["role"] == "assistant":
                assistant_non_loss_token = self.tokenizer.encode(
                    self.assistant_non_loss_token.format_map(
                        {"content": messages[i]["content"]}
                    ),
                    add_special_tokens=False,
                )
                
                input_id += assistant_non_loss_token
                label += [-100] * len(assistant_non_loss_token)

                if i == len(messages)-1 and self.current_model_type in think_model_set and enable_thinking == False :  # 使用 current_model_type
                    think_token = self.tokenizer.encode(
                            self.think_token, 
                            add_special_tokens=False,
                        )
                    input_id += think_token
                    label += [-100] * len(think_token)
                assistant_token = self.tokenizer.encode(
                    self.assistant_token.format_map(
                        {"content": messages[i]["content"]}
                    ),
                    add_special_tokens=False,
                )

                input_id += assistant_token
                label += assistant_token
            elif messages[i]["role"] == "system":
                continue # 上面已经处理过了
            else:
                error_role = messages[i]["role"]
                logger.error(f"未经定义的template类型{error_role}")
                assert False
            # print(input_id)
            # print(label)
            # import pdb
            # pdb.set_trace()

        


        if add_generation_prompt:
            assistant_non_loss_token = self.tokenizer.encode(
                self.assistant_non_loss_token.format_map(
                    {"content": messages[i]["content"]}
                ),
                add_special_tokens=False,
            )
            
            input_id += assistant_non_loss_token
            label += [-100] * len(assistant_non_loss_token)

            if self.current_model_type in think_model_set and enable_thinking == False:  # 使用 current_model_type
                think_token = self.tokenizer.encode(
                        self.think_token, 
                        add_special_tokens=False,
                    )
                input_id += think_token
                label += [-100] * len(think_token)

        if self.efficient_eos:
            if self.end_token_id:
                input_id += [self.end_token_id]
                label += [self.end_token_id]
        return input_id, label


@register_template
class GemmaTemplate(Template):
    model_type = ["gemma", "gemma2","gemma3"]
    # 这个模型系列不支持自定义system，逆天，我自己编一个
    # 从gemma3开始终于半支持了，仍然没有system这个角色！但是现在允许把第一轮的system和user合并了！
    def __init__(self, tokenizer, **kwargs) -> None:

        super().__init__(
            tokenizer=tokenizer,
            # system_token="<start_of_turn>system\n{content}<end_of_turn>\n",
            # default_system="You are a helpful assistant.",
            user_token="<start_of_turn>user\n{content}<end_of_turn>\n",
            assistant_non_loss_token="<start_of_turn>model\n",
            assistant_token="{content}<end_of_turn>\n",
            start_token_id=tokenizer.bos_token_id,
            # end_token_id=tokenizer.eos_token_id,
            efficient_eos=True,
            merge_system_and_first_user=True,
            jinja_template=r"""{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '
' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '
' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '
' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>
' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model
'}}
{%- endif -%}
""",
            **kwargs  # 传递所有额外参数，包括 current_model_type
        )
        self.base_eos_token_id = 1
        self.chat_eos_token_id = 107


@register_template
class Qwen2Template(Template):
    model_type = "qwen2"

    def __init__(self, tokenizer, **kwargs) -> None:
        

        generation_config = {
            "bos_token_id": 151643,
            "pad_token_id": 151643,
            "do_sample": True,
            "eos_token_id": [
            151645,
            151643
            ],
            "repetition_penalty": 1.05,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "transformers_version": "4.37.0"
        }
        

        super().__init__(
            tokenizer=tokenizer,
            user_token="<|im_start|>user\n{content}<|im_end|>\n",
            assistant_non_loss_token="<|im_start|>assistant\n",
            assistant_token="{content}<|im_end|>\n",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=False,
            system_token="<|im_start|>system\n{content}<|im_end|>\n",
            default_system="You are a helpful assistant.",
            generation_config = generation_config,
            **kwargs  # 传递所有额外参数，包括 current_model_type
        )
        # 必须写在后面不然会被默认值覆盖
        self.base_eos_token_id = 151643
        self.chat_eos_token_id = 151645


@register_template
class Qwen25Template(Template):
    model_type = ["qwen2.5"]

    def __init__(self, tokenizer, **kwargs) -> None:
        
        generation_config = {
            "bos_token_id": 151643,
            "pad_token_id": 151643,
            "do_sample": True,
            "eos_token_id": [
            151645,
            151643
            ],
            "repetition_penalty": 1.05,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "transformers_version": "4.37.0"
        }


        super().__init__(
            tokenizer=tokenizer,
            user_token="<|im_start|>user\n{content}<|im_end|>\n",
            assistant_non_loss_token="<|im_start|>assistant\n",
            assistant_token="{content}<|im_end|>\n",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=False,
            system_token="<|im_start|>system\n{content}<|im_end|>\n",
            default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            jinja_template="{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n",
            embedding_size=152064,
            generation_config = generation_config,
            **kwargs  # 传递所有额外参数，包括 current_model_type
        )
        # 必须写在后面不然会被默认值覆盖
        self.base_eos_token_id = 151643
        self.chat_eos_token_id = 151645


@register_template
class Qwen3Template(Template):
    model_type = ["qwen3"]

    def __init__(self, tokenizer, **kwargs) -> None:
        
        generation_config = {
            "bos_token_id": 151643,
            "do_sample": True,
            "eos_token_id": [
                151645,
                151643
            ],
            "pad_token_id": 151643,
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
            "transformers_version": "4.51.0"
        }


        super().__init__(
            tokenizer=tokenizer,
            user_token="<|im_start|>user\n{content}<|im_end|>\n",
            assistant_non_loss_token ="<|im_start|>assistant\n",
            assistant_token="{content}<|im_end|>\n",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=False,
            system_token="<|im_start|>system\n{content}<|im_end|>\n",
            think_token = "<think>\n\n</think>\n\n",
            default_system=None,
            jinja_template="{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}",
            embedding_size=151936,
            generation_config = generation_config,
            **kwargs  # 传递所有额外参数，包括 current_model_type
        )
        # 必须写在后面不然会被默认值覆盖
        self.base_eos_token_id = 151643
        self.chat_eos_token_id = 151645




@register_template
class LlamaTemplate(Template):
    model_type = ["llama"]

    def __init__(self, tokenizer, **kwargs) -> None:

        super().__init__(
            tokenizer=tokenizer,
            user_token="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            assistant_token="{content}<|eot_id|>",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=False,
            **kwargs
        )
        self.base_eos_token_id = 128001
        self.chat_eos_token_id = 128009


@register_template
class Llama2Template(Template):
    model_type = ["llama2"]

    def __init__(self, tokenizer, **kwargs) -> None:

        super().__init__(
            tokenizer=tokenizer,
            user_token="[INST] {content} [/INST]",
            assistant_token="{content}   ",
            start_token_id=None,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=False,
            **kwargs
        )


@register_template
class Llama3Template(Template):
    model_type = ["llama3"]

    def __init__(self, tokenizer,**kwargs) -> None:
        

        today = datetime.date.today()
        formatted_date = today.strftime("%d %b %Y")

        generation_config = {
            "bos_token_id": 128000,
            "eos_token_id": [128001, 128009],
            "do_sample": True,
            "temperature": 0.6,
            "max_length": 4096,
            "top_p": 0.9,
            "transformers_version": "4.40.0.dev0"
        }


        super().__init__(
            tokenizer=tokenizer,
            user_token="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
            assistant_token="{content}<|eot_id|>",
            assistant_non_loss_token="<|start_header_id|>assistant<|end_header_id|>\n\n",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=False,
            system_token="<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
            # default_system=f"""Cutting Knowledge Date: December 2023\nToday Date: {formatted_date}""",
            tool_token=None,
            embedding_size=128256,
            generation_config = generation_config,
            **kwargs
        )
        self.base_eos_token_id = 128001
        self.chat_eos_token_id = 128009



@register_template
class Llama31Template(Template):
    model_type = ["llama3.1"]

    def __init__(self, tokenizer,**kwargs) -> None:


        generation_config = {
            "bos_token_id": 128000,
            "do_sample": True,
            "eos_token_id": [
                128001,
                128008,
                128009
            ],
            "temperature": 0.6,
            "top_p": 0.9,
            "transformers_version": "4.42.3"
            }

        

        today = datetime.date.today()
        formatted_date = today.strftime("%d %b %Y")
        super().__init__(
            tokenizer=tokenizer,
            user_token="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
            assistant_token="{content}<|eot_id|>",
            assistant_non_loss_token="<|start_header_id|>assistant<|end_header_id|>\n\n",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=False,
            system_token="<|start_header_id|>system<|end_header_id|>\n\n{content}\n\n<|eot_id|>",
            default_system=f"""Cutting Knowledge Date: December 2023\nToday Date: {formatted_date}""",
            tool_token=None,
            embedding_size=128256,
            generation_config = generation_config,
            **kwargs
        )
        self.base_eos_token_id = 128001
        self.chat_eos_token_id = 128009



@register_template
class Llama32Template(Template):
    model_type = ["llama3.2"]

    def __init__(self, tokenizer,**kwargs) -> None:
        

        generation_config = {
            "bos_token_id": 128000,
            "do_sample": True,
            "eos_token_id": [
                128001,
                128008,
                128009
            ],
            "temperature": 0.6,
            "top_p": 0.9,
            "transformers_version": "4.45.0.dev0"
            }



        today = datetime.date.today()
        formatted_date = today.strftime("%d %b %Y")
        super().__init__(
            tokenizer=tokenizer,
            user_token="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
            assistant_non_loss_token="<|start_header_id|>assistant<|end_header_id|>\n\n",
            assistant_token="{content}<|eot_id|>",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=False,
            system_token="<|start_header_id|>system<|end_header_id|>\n\n{content}\n\n<|eot_id|>",
            default_system=f"""Cutting Knowledge Date: December 2023\nToday Date: {formatted_date}""",
            tool_token=None,
            embedding_size=128256,
            generation_config = generation_config,
            jinja_template="""{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now(\"%d %b %Y\") %}\n    {%- else %}\n        {%- set date_string = \"26 Jul 2024\" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n        {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n        {{- '\"parameters\": ' }}\n        {{- tool_call.arguments | tojson }}\n        {{- \"}\" }}\n        {{- \"<|eot_id|>\" }}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n""",
            **kwargs
        )
        self.base_eos_token_id = 128001
        self.chat_eos_token_id = 128009





@register_template
class YiTemplate(Template):
    model_type = ["yi"]

    def __init__(self, tokenizer,**kwargs) -> None:

        super().__init__(
            tokenizer=tokenizer,
            user_token="<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
            assistant_token="{content}<|im_end|>\n",
            start_token_id=None,
            end_token_id=None,  # yi的结束词就是<|im_end|>
            efficient_eos=False,
            system_token="<|im_start|>system\n{content}<|im_end|>\n",
            default_system=None,
            **kwargs
        )


@register_template
class MistralTemplate(Template):
    model_type = ["mistral"]

    def __init__(self, tokenizer,**kwargs) -> None:

        super().__init__(
            tokenizer=tokenizer,
            user_token="[INST] {content}[/INST]",
            assistant_token=" {content} ",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=None,
            efficient_eos=True,
            **kwargs
        )


@register_template
class MistralNemoTemplate(Template):
    model_type = ["mistral_nemo"]

    def __init__(self, tokenizer,**kwargs) -> None:

        super().__init__(
            tokenizer=tokenizer,
            user_token="[INST]{content}[/INST]",
            assistant_token="{content} ",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=None,
            efficient_eos=True,
            **kwargs
        )


@register_template
class Phi3Template(Template):
    model_type = ["phi3"]

    def __init__(self, tokenizer,**kwargs) -> None:
        super().__init__(
            tokenizer=tokenizer,
            user_token="<|user|>\n{content}<|end|>\n<|assistant|>\n",
            assistant_token="{content}<|end|>\n",
            start_token_id=None,
            end_token_id=32000,
            efficient_eos=True,
            **kwargs
        )


@register_template
class Phi3SamllTemplate(Template):
    model_type = ["phi3small"]

    def __init__(self, tokenizer,**kwargs) -> None:
        super().__init__(
            tokenizer=tokenizer,
            user_token="<|user|>\n{content}<|end|>\n<|assistant|>\n",
            assistant_token="{content}<|end|>\n",
            start_token_id=tokenizer.eos_token_id,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=True,
            **kwargs
        )




@register_template
class dLLMTEMPLATE(Template):
    model_type = ["dllm"]

    def __init__(self, tokenizer, **kwargs) -> None:
        
        generation_config = {
            "bos_token_id": 50285,
            "do_sample": True,
            "eos_token_id": [
                50294, # sft
                50279, # pretrain
            ],
            "pad_token_id": 50283,
            "temperature": 0.6,
            "top_k": 20,
            "top_p": 0.95,
            "transformers_version": "4.51.0"
        }


        super().__init__(
            tokenizer=tokenizer,
            user_token="[unused8]user\n{content}[unused9]\n",
            assistant_non_loss_token ="[unused8]assistant\n",
            assistant_token="{content}[unused9]\n",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=False,
            system_token="[unused8]system\n{content}[unused9]\n",
            think_token = "<think>\n\n</think>\n\n",
            default_system=None,
            jinja_template="{%- if tools %}\n    {{- '[unused8]system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call>[unused9]\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '[unused8]system\\n' + messages[0].content + '[unused9]\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '[unused8]' + message.role + '\\n' + content + '[unused9]' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '[unused8]' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '[unused8]' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '[unused8]' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '[unused9]\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '[unused8]user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '[unused9]\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '[unused8]assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}",
            embedding_size=50368,
            generation_config = generation_config,
            **kwargs  # 传递所有额外参数，包括 current_model_type
        )
        # 必须写在后面不然会被默认值覆盖
        self.base_eos_token_id = 50279
        self.chat_eos_token_id = 50294


# def test_tool_function():

def test(tokenizer_name, template_key): # Renamed 'template' to 'template_key' to avoid conflict with module
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Ensure the tokenizer has a chat template if apply_chat_template is to work well
    # For many models, this is set by default. For others, you might need to set it manually
    # if it's missing and you want to compare against a specific format.
    # Example for Qwen (already has one, but to illustrate):
   
        

    if template_key not in modelType2Template:
        print(f"Error: Template key '{template_key}' not found in modelType2Template.")
        return False

    g = modelType2Template[template_key](tokenizer)
    if not tokenizer.chat_template:
        tokenizer.chat_template = g.get_chat_template()
    # Scenario 1: User -> Assistant -> User (expecting Assistant response next)
    message1 = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help?"},
        {"role": "user", "content": "Tell me a joke."},
    ]
    
    # Scenario 2: System -> User (expecting Assistant response next)
    message2 = [
        {"role":"system","content":"You are a helpful AI."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    # Scenario 3: User only (expecting Assistant response next)
    message3 = [
        {"role": "user", "content": "Just one user message."},
    ]

    # Scenario 4: User -> Assistant -> User -> Assistant (complete turn, but still add_generation_prompt)
    message4 = [
        {"role": "user", "content": "First user message."},
        {"role": "assistant", "content": "First assistant response."},
        {"role": "user", "content": "Second user message."},
        {"role": "assistant", "content": "Second assistant response. This is the last message in history."},
    ]

    message5 = [
        {"role":"system","content":"You are a helpful AI."},
        {"role": "user", "content": "First user message."},
        {"role": "assistant", "content": "First assistant response."},
    ]
    
    
    messages_to_test = {
        "Scenario1 (UAU)": message1,
        "Scenario2 (SU)": message2,
        "Scenario3 (U)": message3,
        "Scenario4 (UAUA)": message4, # HF apply_chat_template with add_generation_prompt=True will still add assistant prompt
        "Scenario5 (SUA)": message5,
    }

    all_passed = True

    for msg_name, message in messages_to_test.items():
        print(f"\n--- Testing {msg_name} for {tokenizer_name} with template {template_key} ---")

        # `a` is from your custom template, `b` is the text from your custom template (optional)
        a, b_custom_text = g.apply(message)
        xx,_ = g.apply(message, add_generation_prompt=True, enable_thinking=True)
        # `c` is tokenized output from HF's template
        # `add_generation_prompt=True` means it will add the tokens to prompt the model for a response.
        # `enable_thinking=False` (Qwen specific, good default for general comparison)
        try:
            c = tokenizer.apply_chat_template(message, tokenize=True,  enable_thinking=False)
            d_hf_text_no_think = tokenizer.apply_chat_template(message, tokenize=False,  enable_thinking=False)
            e_hf_text_think = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        except Exception as e:
            print(f"Error applying Hugging Face chat template for {tokenizer_name}: {e}")
            print("This might happen if the tokenizer does not have a chat_template defined.")
            print("Or if the message format is incompatible.")
            all_passed = False
            continue

        # Check EOS for HF template (with add_generation_prompt=True, it usually SHOULD NOT end with EOS)
        # This check is more of an observation.
        if c and c[-1] == tokenizer.eos_token_id:
            print(f"INFO: {tokenizer_name} ({msg_name}) - HF template (c) WITH add_generation_prompt=True ENDS with EOS. This is unusual but possible.")
        elif not c :
             print(f"WARNING: {tokenizer_name} ({msg_name}) - HF template (c) is empty.")


        if a != c or (tokenizer.decode(xx) != e_hf_text_think and not msg_name.endswith('A)')):
            all_passed = False
            print("=" * 30)
            print(f"MISMATCH FOUND: {tokenizer_name} ({msg_name})")
            # print(f"Custom Template Tokens (a):\n{a}")
            # print(f"HF Template Tokens (c):\n{c}")
            # print("---------TOKENS AS STRINGS---------")
            # print(f"Custom Tokens (a):\n{tokenizer.convert_ids_to_tokens(a)}")
            # print(f"HF Tokens (c):\n{tokenizer.convert_ids_to_tokens(c)}")
            print("---------DECODED STRINGS---------")
            print(f"Custom Decoded (from a):\n>>>\n{tokenizer.decode(a)}\n<<<")
            print(f"HF Decoded (from c) / HF String (d):\n>>>\n{d_hf_text_no_think}\n<<<")
            if b_custom_text: # If your custom template also returns its string version
                 print(f"Custom Text (b_custom_text):\n>>>\n{b_custom_text}\n<<<")
            print("=" * 30)
            import pdb
            pdb.set_trace()
        else:
            print(f"SUCCESS: {tokenizer_name} ({msg_name}) - Custom template matches Hugging Face template.")
            # print(f"HF Text (d):\n{d_hf_text_no_think}")
            # print(f"Custom Decoded (a):\n{tokenizer.decode(a)}")


    return all_passed

if __name__ == "__main__":
    # from config import model_dir

    # 测试要拿IT版本测！这样才能对齐
    test_list = [
        # ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
        # ("mistralai/Mistral-Nemo-Instruct-2407", "mistral_nemo"),
        # ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
        # ("Qwen/Qwen1.5-32B-Chat", "qwen2"),
        # ("microsoft/Phi-3-small-8k-instruct", "phi3small"),
        # ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
        # ("google/gemma-7b-it", "gemma2"),
        # ("unsloth/gemma-2b-it", "gemma2"),
        # ("unsloth/llama-3-8b-Instruct","llama3")
        # (model_dir['llama3.1-8b-it'], "llama3.1"),
        # ("meta-llama/Llama-2-7b-chat-hf", "llama2"),
        # ("01-ai/Yi-1.5-34B-Chat", "yi"),
        # (model_dir['gemma2-27b-it'], "gemma2"),
        # ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5"),
        # (model_dir['gemma3-4b-it'], "gemma3"),
        # ("google/gemma-3-270m",'gemma3')
        ("unsloth/Llama-3.2-3B-Instruct", "llama3.2"),
        # ("Qwen/Qwen3-4B","qwen3"),
        # ("/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen3-235B-A22B-Instruct-2507/main","qwen3"),
        # ("unsloth/gemma-3-27b-it","gemma3"),
        # ("Qwen/Qwen3-4B-Instruct-2507","qwen3")
    ]
    # TODO 现在在思维模式下，如果传进来的messages最后一个是assistant，hf的行为是当做nothink去处理。
    # 但是如果我们训练希望嵌入思维链，就需要思考怎么做，我们可以把这个当成think的内容插入进去
    result = {}
    for instance in test_list:
        if not test(instance[0], instance[1]):
            result[instance[0]] = "fail"
        else:
            result[instance[0]] = "success"
    print(result)