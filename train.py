import torch
import numpy as np
import argparse
import time
import util

from engine import trainer
from event_logger import setup_logger

def main(args):
    
    util.make_dir_if_not_exist(args.save)
    
    logger = setup_logger(name="default", log_file="{}/experiment.log".format(args.save))
    
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    engine = trainer(scaler, args.in_dim, args.seq_length, args.nhid, args.dropout, args.learning_rate, args.weight_decay, args.device, supports)
    logger.info("start training...")
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator(), start=1):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                logger.info(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]))
        t2 = time.time()
        train_time.append(t2-t1)

        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        logger.info(log.format(i, (s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}\nTrain Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}\nValid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}\nTraining Time: {:.4f}/epoch'
        logger.info(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)))

        
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss, 2))+".pth")

    logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid], 2))+".pth"))

    outputs = []
    realy = []
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testy = torch.Tensor(y).to(device)
        testy = testy.transpose(1, 3)[:, :1, :, :]
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds)
        realy.append(testy)

    yhat = torch.cat(outputs, dim=0)
    yhat = scaler.inverse_transform(yhat)
    realy = torch.cat(realy, dim=0)

    logger.info("Training finished")
    logger.info("The valid loss on best model is {}".format(np.round(his_loss[bestid], 4)))

    amae = []
    amape = []
    armse = []
    logger.info("The shape of predicted {} and label {}".format(yhat.shape, realy.shape))
    for i in range(12):
        pred = yhat[..., i]
        real = realy[..., i]
        metrics = util.metric(pred, real)
        logger.info('Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    logger.info(
        'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'.format(*util.metric(yhat, realy))
    )
    torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid], 2))+".pth")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda:1', help='')
    parser.add_argument('--data', type=str, default='data/metr', help='data path')
    parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
    parser.add_argument('--seq_length', type=int, default=12, help='')
    parser.add_argument('--nhid', type=int, default=64, help='')
    parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--epochs', type=int, default=30, help='')
    parser.add_argument('--print_every', type=int, default=1000, help='')
    parser.add_argument('--save', type=str, default='scratch/exp4/', help='save path')
    parser.add_argument('--expid', type=int, default=1, help='experiment id')

    args = parser.parse_args()
    
    t1 = time.time()
    main(args)
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
