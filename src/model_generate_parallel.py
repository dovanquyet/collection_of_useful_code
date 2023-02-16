from itertools import chain
import threading
import torch
from torch.cuda._utils import _get_device_index
from torch.cuda.amp import autocast
from torch._utils import ExceptionWrapper


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def parallel_apply_generate(modules, inputs, kwargs_tup=None, devices=None):
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.
    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices
    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = [_get_device_index(x, True) for x in devices]
    streams = [torch.cuda.current_stream(x) for x in devices]
    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

    def _worker(i, module, input, kwargs, device=None, stream=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        if stream is None:
            stream = torch.cuda.current_stream(device)
        try:
            with torch.cuda.device(device), torch.cuda.stream(stream), autocast(enabled=autocast_enabled):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module.generate(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device, stream))
                   for i, (module, input, kwargs, device, stream) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices, streams))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0], streams[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs


class ModelGenerateParallel(torch.nn.DataParallel):
  def __init__(self, model, device_ids):
    super(ModelParallel, self).__init__(model, device_ids)

  def forward(self, *inputs, **kwargs):
    if not self.device_ids:
        return self.module.generate(*inputs, **kwargs)

    for t in chain(self.module.parameters(), self.module.buffers()):
        if t.device != self.src_device_obj:
            raise RuntimeError("module must have its parameters and buffers "
                               "on device {} (device_ids[0]) but found one of "
                               "them on device: {}".format(self.src_device_obj, t.device))

    inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
    if len(self.device_ids) == 1:
        return self.module.generate(*inputs[0], **kwargs[0])

    replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
    outputs = parallel_apply_generate(replicas, inputs, kwargs)

    return self.gather(outputs, self.output_device)


def main():
    from transformers import GPT2LMHeadModel, AutoTokenizer, set_seed
    import pandas as pd

    # load model from transformers
    num_gpus = torch.cuda.device_count()
    model_name_or_path = '../checkpoint/models/concept_generator'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path).eval().to('cuda')
    set_seed(0)

    # parallelize the model
    from model_generate_parallel import ModelGenerateParallel
    model = ModelParallel(model, list(range(num_gpus)))

    # setting up
    n_samples, batch_size = 1000, 32
    num_steps = (len(df)-1)//batch_size + 1
    n_return_seq, len_for_gen = 1, 4
    df = pd.DataFrame({'inputs': ['Hello world']*n_samples}) # NOTE: load your data here
    outputs = []

    # generation
    print('Start generating. Num of gpus used is', num_gpus)
    for i in tqdm.tqdm(range(num_steps)):
        inputs = df['inputs'][i*batch_size: (i+1)*batch_size].tolist()
        input_ids = tokenizer(inputs, return_tensors='pt', 
            padding='longest', truncation=True).to('cuda')
        generated_ids = model(
            **input_ids,  # when len_for_gen=3, wow, using 4 gpus is just 1.8h, while using 2 or 8 gpus needs 2h to run
            output_scores=True, early_stopping=True, max_new_tokens=len_for_gen,  # faster than old setting, 2.5h vs 3.5h
            num_return_sequences=n_return_seq, num_beams=n_return_seq+1,
            top_p=0.9, top_k=5, repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        outputs.extend(generated_text)


if __name__ == '__main__':
    main()