from autoattack import AutoAttack
import torch
import torchvision

if args.attack == 'AA':
        
    test_set = val_dataset        
    data = [test_set[i][0] for i in range(0, 5000)]
    data = torch.stack(data).cuda()
    target = [test_set[i][1] for i in range(0, 5000)]
    target = torch.LongTensor(target).cuda()
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
            version='standard')
    adv_complete = adversary.run_standard_evaluation(data, target, bs=args.batch_size)
        
    import torchattacks as attack

if args.attack == 'PGD':
    if args.norm == 'L2':
        adversary = attack.PGDL2(model, eps=args.epsilon, alpha=0.1, steps=args.step)
    else:
        adversary = attack.PGD(model, eps=args.epsilon, alpha=1/255, steps=args.step)
else:
    adversary = attack.CW(model, steps=args.step)

adversary = attack.PGDL2(model, eps=0.5, alpha=0.1, steps=args.step)
cnt, correct = 0, 0
for image, label in tqdm(val_dataloader):
    adv_img = adversary(image.cuda(), label.cuda())
    logits = model(adv_img)
    correct += (logits.argmax(axis=1).cpu() == label).sum().cpu().item()
    
    print(correct, image.shape[0])
    
    cnt += 1
    if cnt == 20:
        break

print(f'robust acc = {correct/5000}')


adversary = attack.PGD(model, eps=0.0157, alpha=1/255, steps=args.step)
cnt, correct = 0, 0
for image, label in tqdm(val_dataloader):
    adv_img = adversary(image.cuda(), label.cuda())
    logits = model(adv_img)
    correct += (logits.argmax(axis=1).cpu() == label).sum().cpu().item()
    
    print(correct, image.shape[0])
    
    cnt += 1
    if cnt == 20:
        break

print(f'robust acc = {correct/5000}')