    
train_dataset = VQA_Slake_Dataset(data_args, tokenizer, mode='train', knowledge_encoder=model_args.knowledge_encoder)
eval_dataset = VQA_Slake_Dataset(data_args, tokenizer, mode='validation', knowledge_encoder=model_args.knowledge_encoder) #CapDataset(data_args, tokenizer, mode='validation')
test_dataset = VQA_Slake_Dataset(data_args, tokenizer, mode='test', knowledge_encoder=model_args.knowledge_encoder) #CapDataset(data_args, tokenizer, mode='validation')
data_collator = DataCollator(data_args.seg_enable)

if(not training_args.eval_only):
    rank0_print("="*20 + " Training " + "="*20)

    trainer = BaMCoVQATrainer(
                            model=model,
                            args=training_args,
                            data_collator=data_collator,
                            train_dataset=train_dataset,
                            eval_dataset=eval_dataset,
                            processing_class=tokenizer,
                            callbacks=[transformers.EarlyStoppingCallback(4)],
                    )

    print(trainer.place_model_on_device)
    
    
    trainer.train()

    rank0_print("="*20 + " Saving the Best Model " + "="*20)
    torch.save(model.state_dict(), os.path.join(training_args.checkpoint_dir, 'pytorch_model_best3.bin'))