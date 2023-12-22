from transformers import Seq2SeqTrainer

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # 调用模型，并获取输出
        outputs = model(**inputs)
        logits = outputs.logits

        # 定义您的自定义损失函数
        # 例如，使用交叉熵损失
        labels = inputs["labels"]
        # 忽略 -100 的标签，这些是填充的部分
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# 然后在训练代码中使用这个自定义的 Trainer
# model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained)
# trainer = CustomSeq2SeqTrainer(model=model, **trainer_kwargs)
# trainer.train()


'''
注意：compute_metrics 和 compute_loss 在 Trainer 类或其子类（如 Seq2SeqTrainer）中扮演不同的角色，它们是两种不同的参数或功能：

compute_metrics：

作用：compute_metrics 用于在模型的评估阶段计算和报告性能指标。它通常用于计算准确率、F1 分数、精确度、召回率等指标。
如何使用：您可以将一个自定义的函数作为 compute_metrics 参数传递给 Trainer 类。这个函数需要接收一个包含模型预测和对应标签的参数，并返回一个包含计算出的指标的字典。
compute_loss：

作用：compute_loss 方法定义了在模型训练过程中如何计算损失。这对于自定义训练过程中的损失函数特别重要，尤其是当标准损失函数不适用于您的特定任务时。
如何使用：要自定义损失函数，您需要继承 Trainer 类或其子类，并重写 compute_loss 方法。这个方法需要接收模型和输入，并返回计算出的损失。
简而言之，compute_metrics 是在模型评估时使用的，用于计算性能指标；而 compute_loss 是在模型训练时使用的，用于计算损失。这两者在训练和评估模型时都非常重要，但它们服务于不同的阶段和目的。
'''