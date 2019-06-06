class RANSAC {
    constructor() {
        this.modelPoints = 0;
        this.threshold = 0;
        this.confidence = 0.99;
        this.maxIters = 1000;
        this.maxAttempts = 1000;
        this.cb = new PointSetRegistrator();
    }

    genMatrix() {
    }

    checkVector() {
    }

    getSubset(m1, m2, ms1, ms2, rng, ..) {
    }

    findInliers() {
    }
    
    rowRange() {
    }

    clear(arr) {
        arr.splice(0, arr.length);
    }

    need(src) {
        return true;
    }

    RANSACUpdateNumIters(p, ep, modelPoints, maxIters) {
        if (modelPoints <= 0) {
            return;
        }
        p = Math.max(p, 0.0);
        p = Math.min(p, 1.0);
        ep = Math.max(ep, 0.0);
        ep = Math.min(ep, 1.0);
        let num = Math.max(1.0 - p, DBL_MIN);
        let denom = 1.0 - Math.pow(1.0 - ep, modelPoints);
        if (denom < DBL_MIN) {
            return 0;
        }
        num = Math.log(num);
        denom = Math.log(denom);

        return denom >= 0 || -num >= maxIters * (denom) ? maxIters : Math.round(num / denom); 
    }

    run(m1, m2, model, mask) {
        let result = false;
        let iter = 0;
        let niters = Math.max(this.maxIters, 1);
        let d1 = m1[0].length; 
        let d2 = m2[0].length; 
        let count1 = this.checkVector(m1, d1);
        let count2 = this.checkVector(m2, d2);
        let maxGoodCount = 0;
        let rng = new RNG();
        if (count1 < modelPoints) {
            return false;
        }

        let bestMask0 = [];
        let bestMask = [];

        if (this.need(mask)) {
            mask = this.genMatrix(count1, 1, -1);
            bestMask0 = this.genMatrix(count1, 1, -1);
            bestMask = this.genMatrix(count1, 1, -1);
        } else {
            bestMask = this.genMatrix(count1, 1, 0);
            bestMask0 = this.genMatrix(count1, 1, 0);
        }
        if (count1 === this.modelPoints) {
            if (cb.runKernel(m1, m2, bestModel) <= 0) {
                return false;
            }
            model = bestModel;
            bestMask = this.genMatrix(count1, 1, 1);
            return true;
        }

        for (iter = 0; iter < niters; iter++) {
            let i = 0;
            let nmodels = 0;
            if (count > modelPoints) {
                let found = this.getSubset(m1, m2 , ms1, ms2, rng, 10000);
                if (!found) {
                    if (iter === 0) {
                        return false;
                    }
                    break;
                }
            }

            nmodels = cb.runKernel(ms1, ms2, model);
            if (nmodels <= 0) {
                continue;
            }
            // OpenCV Size {width, height}
            let modelSize = [model[0].length, model.length];
            for (i = 0; i < nmodels; i++) {
                let model_ = this.rowRange(i * modelSize[1], (i + 1) * modelSize[1]);
                let goodCount = this.findInliers(m1, m2, model_, err, mask, threshold);
                if (goodCount > Math.max(maxGoodCount, modelPoints -1)) {
                    // this.swap(mask, bestMask);
                    let tempMask = mask;
                    mask = bestMask;
                    bestMask = tempMask;
                    bestModel = model_;
                    maxGoodCount = goodCount;
                    niters = this.RANSACUpdateNumIters(confidence, (count - goodCount) / count, modelPoints, niters);
                }
            }
        }

        if (maxGoodCount > 0) {
            /*
            // if( bestMask != bestMask0) {
            if(!this.matEqual(bestMask, bestMask0)) {
                if (bestMask.length === bestMask0.length && bestMask[0].length === bestMask0[0].length) {
                    bestMask0 = bestMask;
                } else {
                    this.tranpose(bestMask, bestMask0);
                }
            }
            */
            model = bestModel;
            rersult = true;
        } else {
            this.clear(model);
        }
        return result;
    }
    
}



class RANSACPointSetRegistrator : public PointSetRegistrator
{
public:
    RANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb=Ptr<PointSetRegistrator::Callback>(),
                              int _modelPoints=0, double _threshold=0, double _confidence=0.99, int _maxIters=1000)
      : cb(_cb), modelPoints(_modelPoints), threshold(_threshold), confidence(_confidence), maxIters(_maxIters) {}

    int findInliers( const Mat& m1, const Mat& m2, const Mat& model, Mat& err, Mat& mask, double thresh ) const
    {
        cb->computeError( m1, m2, model, err );
        mask.create(err.size(), CV_8U);

        CV_Assert( err.isContinuous() && err.type() == CV_32F && mask.isContinuous() && mask.type() == CV_8U);
        const float* errptr = err.ptr<float>();
        uchar* maskptr = mask.ptr<uchar>();
        float t = (float)(thresh*thresh);
        int i, n = (int)err.total(), nz = 0;
        for( i = 0; i < n; i++ )
        {
            int f = errptr[i] <= t;
            maskptr[i] = (uchar)f;
            nz += f;
        }
        return nz;
    }

    bool getSubset( const Mat& m1, const Mat& m2,
                    Mat& ms1, Mat& ms2, RNG& rng,
                    int maxAttempts=1000 ) const
    {
        cv::AutoBuffer<int> _idx(modelPoints);
        int* idx = _idx.data();
        int i = 0, j, k, iters = 0;
        int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
        int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
        int esz1 = (int)m1.elemSize1()*d1, esz2 = (int)m2.elemSize1()*d2;
        int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
        const int *m1ptr = m1.ptr<int>(), *m2ptr = m2.ptr<int>();

        ms1.create(modelPoints, 1, CV_MAKETYPE(m1.depth(), d1));
        ms2.create(modelPoints, 1, CV_MAKETYPE(m2.depth(), d2));

        int *ms1ptr = ms1.ptr<int>(), *ms2ptr = ms2.ptr<int>();

        CV_Assert( count >= modelPoints && count == count2 );
        CV_Assert( (esz1 % sizeof(int)) == 0 && (esz2 % sizeof(int)) == 0 );
        esz1 /= sizeof(int);
        esz2 /= sizeof(int);

        for(; iters < maxAttempts; iters++)
        {
            for( i = 0; i < modelPoints && iters < maxAttempts; )
            {
                int idx_i = 0;
                for(;;)
                {
                    idx_i = idx[i] = rng.uniform(0, count);
                    for( j = 0; j < i; j++ )
                        if( idx_i == idx[j] )
                            break;
                    if( j == i )
                        break;
                }
                for( k = 0; k < esz1; k++ )
                    ms1ptr[i*esz1 + k] = m1ptr[idx_i*esz1 + k];
                for( k = 0; k < esz2; k++ )
                    ms2ptr[i*esz2 + k] = m2ptr[idx_i*esz2 + k];
                i++;
            }
            if( i == modelPoints && !cb->checkSubset(ms1, ms2, i) )
                continue;
            break;
        }

        return i == modelPoints && iters < maxAttempts;
    }

    bool run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _mask) const CV_OVERRIDE
    {
        bool result = false;
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        Mat err, mask, model, bestModel, ms1, ms2;

        int iter, niters = MAX(maxIters, 1);
        int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
        int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
        int count = m1.checkVector(d1), count2 = m2.checkVector(d2), maxGoodCount = 0;

        RNG rng((uint64)-1);

        CV_Assert( cb );
        CV_Assert( confidence > 0 && confidence < 1 );

        CV_Assert( count >= 0 && count2 == count );
        if( count < modelPoints )
            return false;

        Mat bestMask0, bestMask;

        if( _mask.needed() )
        {
            _mask.create(count, 1, CV_8U, -1, true);
            bestMask0 = bestMask = _mask.getMat();
            CV_Assert( (bestMask.cols == 1 || bestMask.rows == 1) && (int)bestMask.total() == count );
        }
        else
        {
            bestMask.create(count, 1, CV_8U);
            bestMask0 = bestMask;
        }

        if( count == modelPoints )
        {
            if( cb->runKernel(m1, m2, bestModel) <= 0 )
                return false;
            bestModel.copyTo(_model);
            bestMask.setTo(Scalar::all(1));
            return true;
        }

        for( iter = 0; iter < niters; iter++ )
        {
            int i, nmodels;
            if( count > modelPoints )
            {
                bool found = getSubset( m1, m2, ms1, ms2, rng, 10000 );
                if( !found )
                {
                    if( iter == 0 )
                        return false;
                    break;
                }
            }

            nmodels = cb->runKernel( ms1, ms2, model );
            if( nmodels <= 0 )
                continue;
            CV_Assert( model.rows % nmodels == 0 );
            Size modelSize(model.cols, model.rows/nmodels);

            for( i = 0; i < nmodels; i++ )
            {
                Mat model_i = model.rowRange( i*modelSize.height, (i+1)*modelSize.height );
                int goodCount = findInliers( m1, m2, model_i, err, mask, threshold );

                if( goodCount > MAX(maxGoodCount, modelPoints-1) )
                {
                    std::swap(mask, bestMask);
                    model_i.copyTo(bestModel);
                    maxGoodCount = goodCount;
                    niters = RANSACUpdateNumIters( confidence, (double)(count - goodCount)/count, modelPoints, niters );
                }
            }
        }

        if( maxGoodCount > 0 )
        {
            if( bestMask.data != bestMask0.data )
            {
                if( bestMask.size() == bestMask0.size() )
                    bestMask.copyTo(bestMask0);
                else
                    transpose(bestMask, bestMask0);
            }
            bestModel.copyTo(_model);
            result = true;
        }
        else
            _model.release();

        return result;
    }

    void setCallback(const Ptr<PointSetRegistrator::Callback>& _cb) CV_OVERRIDE { cb = _cb; }

    Ptr<PointSetRegistrator::Callback> cb;
    int modelPoints;
    double threshold;
    double confidence;
    int maxIters;
};

