import numpy as np
from collections import namedtuple

SegMetric = namedtuple('SegMetric','precision recall fscore IoU')
class SumPixel(object):
    def __init__(self,c11,p1,g1):
        self.c11 = c11
        self.p1 = p1
        self.g1 = g1

class MeasureEvaluation(object):
    def __init__(self,num_class=2):
        self.nopixel = np.asarray([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        self.num_class = num_class
        self.sum_pixel = []
        for i in range(self.num_class):
            self.sum_pixel.append(SumPixel(0,0,0))

    def calMetricPerClassPerImage(self,pred,gt):
        if not pred.shape== gt.shape:
            pred = np.argmax(pred, axis=0)
        num_pixels = pred.shape[0]*pred.shape[1]
        metrics = []
        mean_acc = 0.0
        mIoU = 0.0
        sum_c11 = 0
        for idx in range(self.num_class):
            p1 = np.asarray(pred==idx,dtype=np.uint8)
            g1 = np.asarray(gt==idx,dtype=np.uint8)
            c11 = np.asarray(p1*g1,dtype=np.uint8)
            precision = 1.0*np.sum(c11)/np.sum(p1)
            recall = 1.0*np.sum(c11)/np.sum(g1)
            fscore = 2.0*precision*recall/(precision+recall)
            IoU = 1.0*np.sum(c11) / (np.sum(g1) + np.sum(p1) - np.sum(c11))
            metrics.append(SegMetric(precision,recall,fscore,IoU))
            mean_acc += precision
            mIoU += IoU
            self.sum_pixel[idx].c11 += np.sum(c11)
            self.sum_pixel[idx].g1 += np.sum(g1)
            self.sum_pixel[idx].p1 += np.sum(p1)
            sum_c11 += np.sum(c11)
            print('label: ',idx,", precision: %f recall: %f fscore: %f IoU: %f" %metrics[idx])
        c11 = np.asarray(pred==gt,dtype=np.uint8)
        assert(np.sum(c11)==sum_c11)
        over_pre = 1.0*np.sum(c11)/num_pixels
        mean_acc = 1.0/self.num_class * mean_acc
        mIoU = 1.0/self.num_class * mIoU
        print('Overall Precision: ', over_pre)
        print('Mean  Accuracy: ', mean_acc)
        print('Mean IoU: ',mIoU)

    def calMetricOverall(self):
        print('Overall:')
        g_c11 = 0
        g_g1 = 0
        g_p1 =0
        pc =0
        ji =0
        metrics=[]
        for idx in range(self.num_class):
            p1 = self.sum_pixel[idx].p1
            g1 = self.sum_pixel[idx].g1
            c11 = self.sum_pixel[idx].c11
            precision = 1.0*c11/p1
            recall = 1.0 * c11/g1
            fscore=2.0*precision*recall/(precision+recall)
            IoU = c11/(p1+g1-c11)
            print('class ',idx,'  precision: %f, recall: %f, fscore: %f, IoU: %f' %(precision,recall,fscore,IoU))
            metrics.append(fscore)
            g_c11 += c11
            g_g1 += g1
            pc += (1.0*c11/g1)
            ji += (1.0*c11/(g1+p1-c11))
        overall_precision = 1.0*g_c11/g_g1
        per_class_rate = 1.0/self.num_class*pc
        ji = 1.0/self.num_class*ji
        print('Overall Precision: ', overall_precision)
        metrics.append(('op',overall_precision))
        print('Mean  Accuracy: ', per_class_rate)
        metrics.append(('mean_accuracy',per_class_rate))
        print('Mean IoU: ',ji)
        metrics.append(('miou',ji))
        return metrics

    def calMetric(self):
        print('total:', self.nopixel)
        result = {}
        result['precision'] = 1.0 * self.nopixel[4] / self.nopixel[0]
        result['recall'] = 1.0 * self.nopixel[4] / self.nopixel[2]
        result['falseAlarm'] = 1.0 * self.nopixel[6] / self.nopixel[0]
        result['fscore'] = 2 * result['precision'] * result['recall'] / (result['precision'] + result['recall'])
        result['op'] = 1.0 * (self.nopixel[7] + self.nopixel[4]) / (self.nopixel[2] + self.nopixel[3])
        result['ma'] = 0.5 * (1.0 * self.nopixel[7] / self.nopixel[3] + 1.0 * self.nopixel[4] / self.nopixel[2])
        result['ji'] = 0.5 * (
            self.nopixel[7] / (self.nopixel[3] + self.nopixel[1] - self.nopixel[7]) + self.nopixel[4] / (self.nopixel[2] + self.nopixel[0] - self.nopixel[4]))
        for k, v in result.items():
            print(k, v)
        return result
