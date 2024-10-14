import os
import paddle
import math
import traceback
from .adabound import AdaBound
from .audio import Audio
from .evaluation import validate
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder


def train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger,
    hp, hp_str):
    embedder_pt = paddle.load(path=str(args.embedder_path))
    embedder = SpeechEmbedder(hp).cuda()
    embedder.set_state_dict(state_dict=embedder_pt)
    embedder.eval()
    audio = Audio(hp)
    model = VoiceFilter(hp).cuda()
    if hp.train.optimizer == 'adabound':
        optimizer = AdaBound(model.parameters(), lr=hp.train.adabound.
            initial, final_lr=hp.train.adabound.final)
    elif hp.train.optimizer == 'adam':
        optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
            learning_rate=hp.train.adam, weight_decay=0.0)
    else:
        raise Exception('%s optimizer not supported' % hp.train.optimizer)
    step = 0
    if chkpt_path is not None:
        logger.info('Resuming from checkpoint: %s' % chkpt_path)
        checkpoint = paddle.load(path=str(chkpt_path))
        model.set_state_dict(state_dict=checkpoint['model'])
        optimizer.set_state_dict(state_dict=checkpoint['optimizer'])
        step = checkpoint['step']
        if hp_str != checkpoint['hp_str']:
            logger.warning('New hparams is different from checkpoint.')
    else:
        logger.info('Starting new training run')
    try:
        criterion = paddle.nn.MSELoss()
        while True:
            model.train()
            for dvec_mels, target_mag, mixed_mag in trainloader:
                target_mag = target_mag.cuda(blocking=True)
                mixed_mag = mixed_mag.cuda(blocking=True)
                dvec_list = list()
                for mel in dvec_mels:
                    mel = mel.cuda(blocking=True)
                    dvec = embedder(mel)
                    dvec_list.append(dvec)
                dvec = paddle.stack(x=dvec_list, axis=0)
                dvec = dvec.detach()
                mask = model(mixed_mag, dvec)
                output = mixed_mag * mask
                loss = criterion(output, target_mag)
                optimizer.clear_gradients(set_to_zero=False)
                loss.backward()
                optimizer.step()
                step += 1
                loss = loss.item()
                if loss > 100000000.0 or math.isnan(loss):
                    logger.error('Loss exploded to %.02f at step %d!' % (
                        loss, step))
                    raise Exception('Loss exploded')
                if step % hp.train.summary_interval == 0:
                    writer.log_training(loss, step)
                    logger.info('Wrote summary at step %d' % step)
                if step % hp.train.checkpoint_interval == 0:
                    save_path = os.path.join(pt_dir, 'chkpt_%d.pt' % step)
                    paddle.save(obj={'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'step': step,
                        'hp_str': hp_str}, path=save_path)
                    logger.info('Saved checkpoint to: %s' % save_path)
                    validate(audio, model, embedder, testloader, writer, step)
    except Exception as e:
        logger.info('Exiting due to exception: %s' % e)
        traceback.print_exc()
