class Stitcher {
    constructor() {
        this.regRes = 0.6;
        this.seamEstRes = 0.1;
        this.composeRes = -1.0;
        this.confidenceThresh = 1;
        this.doWaveCorrect = false;
        this.workScale = 1;
        this.seamScale = 1;
        this.seamWorkAspect = 1;
        this.warpedImgScale = 1;
        this.imgs = [];
        this.masks = [];
        // { heidht, width }
        this.fullImgSizes = [];
        this.features = [];
        this.pairMatches = [];
        this.seamEstImgs = [];
        this.indices = [];
        this.cameras = [];
        this.resultMask = [];
        this.estimator = new what(); 
        this.bundleAdjuster = new what(); 
    }

    genMatrix(height, width, val) {
        if (height === undefined || height < 1 || width < 1) {
            return [];
        } else if (width === undefined) {
            let arr1D = [];
            for (let i = 0; i < height; i++) {
                arr1D.push(val);
            }
            return arr1D;
        } else {
            let arr1D = [];
            let arr2D = [];
            for (let i = 0; i < width; i++) {
                arr1D.push(val);
            }
            for (let i = 0; i < height; i++) {
                arr2D.push(arr1D.slice());
            }
            return arr2D;
        }
    }

    resize(mat) {
    }

    clear(arr) {
        arr.splice(0, arr.length);
    }

    composePanorama(images, pano) {
        const height = images[0].length;
        const width = images[0][0].length;
        // let img = this.genMatrix(height, width, 0);
        let img = [];
        this.resize(this.seamEstImgs);
        for (let i = 0; i < images.length; i++) {
            this.imgs[i] = images[i];
            img = this.resize(this.imgs[i], this.seamScale, this.seamScale);
            this.seamEstImgs[i] = img;
        }
        let subSeamEstImgs = [];
        let subImgs = [];
        for (let i = 0; i < this.indices.length; i++) {
            subImgs.push(this.imgs[indices[i]]);
            subSeamEstImgs.push(this.seamEstImgs[indices[i]]);
        }
        this.seamEstImgs = subSeamEstImgs;
        this.imgs = subImgs;

        let pano_ = [];
        const numImages = this.imgs.length;
        let corners = this.genMatrix(numImages);
        let imagesWarped = this.genMatrix(numImages);
        let masksWarped = this.genMatrix(numImages);
        let sizes = this.genMatrix(numImages);
        let masks = [];
        for (let i = 0; i < numImages; i++) {
            masks.push(this.genMatrix(this.seamEstImgs[i].length, this.seamEstImgs[i][0].length, 255)); 
        }

        let w = new Warper(this.warpedImgScale * this.seamWorkAspect);
        for (let i = 0; i < numImages; i++) {
            let K = this.cameras[i].K();
            K[0][0] *= this.seamWorkAspect;
            K[0][2] *= this.seamWorkAspect;
            K[1][1] *= this.seamWorkAspect;
            K[1][2] *= this.seamWorkAspect;

            const seamEstImgSize = {
                height: this.seamEstImgs[i].length,
                width: this.seamEstImgs[i][0].length
            };
            corners[i] = w.warp(seamEstImgSize, K, this.cameras[i].R, interpFlags, BORDER_REFLECT, imagesWarped[i]);
            sizes[i] = imagesWarped[i].size;
            const maskSize = {
                height: this.masks[i].length,
                width: this.masks[i][0].length
            };
            w.warp(maskSize, K, this.cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masksWarped[i]);
        }

        this.exposureComp.feed(conners, imagesWarped, masksWarped);
        for (let i = 0; i < numImages; i++) {
            this.exposureComp.apply(i, corners[i], imagesWarped[i], masksWarped[i]);
        }
        seamFinder.find(imagesWarped, corners, masksWarped);
        
        this.clear(this.seamEstImgs);
        this.clear(imagesWarped);
        this.clear(masks);
        

        let imgWarped = []; 
        let composeWorkAspect = 1;
        let isBlenderPrepared = false;
        let composeScale = 1;
        let isComposeScaleSet = false;
        let  camerasScaled = [];

        for (let idx = 0; idx < numImages; idx++) {
            let fullImg = this.imgs[idx];
            if (!isComposeScaleSet) {
                if (this.composeRes > 0) {
                    composeScale = Math.min(1.0, Math.sqrt(this.composeRes * 1e6 / fullImg.length * fullImg[0].length));
                }
                isComposeScaleSet = true;
                composeWorkAspect = composeScale / this.workScale;
                let warpScale = this.wapredImgScale * composeWorkAspect;
                w = new warp(warpScale);

                for (let i = 0; i < this.imgs.length; i++) {
                    camerasScaled[i].ppx *= composeWorkAspect;
                    camerasScaled[i].ppy *= composeWorkAspect;
                    camerasScaled[i].focal *= composeWorkAspect;

                    let sz = this.fullImgSizes[i];
                    if (Math.abs(composeScale - 1) > 0.1) {
                        sz.height = Math.round(this.fullImgSizes[i].height * composeScale);
                        sz.width = Math.round(this.fullImgSizes[i].width * composeScale);
                    }
                    let K = camerasScaled[i].K();
                    let roi = w.warpROI(sz, K, camerasScaled[i].R);
                    corners[i] = roi.tl();
                    sizes[i] = roi.size();
                }
            }
            if (Math.abs(composeScaled - 1) > 0.1) {
                this.resize(fullImg, img, composeScale, composeScale, INTER_LINEAR_EXACT);
            } else {
                img = fullImg;
            }
            this.clear(fullImg);
            let K = camerasScaled[idx].K();
            w.warp(img, K, this.cameras[idx].R, this.interpFlags, BORDER_REFLECT, imgWarped); 
            mask = this.genMatrix(img.length, img[0].length, 255);
            w.warp(mask, K, this.cameras[idx].R, INTER_NEAREST, BORDER_REFLECT, imgWarped); 
            this.clear(imgWarped);
            this.clear(mask);
            this.dilate(masksWarped[idx], dilatedMask);
            this.resize(dilatedMask, seamMask, maskWarped.length,  maskWarped[0].length); 
            this.bitwiseAnd(seamMask, maskWarped, maskWarped);
            if ( !isBlenderPrepared) {
                this.blender.prepare(cornes, sizes);
                isBlenderPrepared = true;
            }
            this.blender.feed(imgWarped, maskWarped, corners[idx]);
            blender.blend(pano, this.resultMask);
            return OK;
        }
    }

    stitch(images, masks, pano) {
        const status = this.estimateTransform(images, masks);
        if (!status) {
            return status;
        }
        return this.composePanorama(pano);
    }

    estimateTransform(images, masks) {
        this.imgs = images;
        this.masks = masks;
        let status = 0;
        if ((status = this.matchImages()) !== OK) {
            return status;
        }
        if ((status = this.estimateCameraParams()) !== OK) {
            return status;
        }
        return OK;
    }

    matchImages() {
        if (this.imgs.length < 2) {
            return ERR_NEED_MORE_IMGS;
        }

        const numImages = this.imgs.length;
        const height = this.imgs[0].length;
        const width = this.imgs[0][0].length;
        this.workScale = 1;
        this.seamWorkAspect = 1;
        this.seamScale = 1
        let isWorkScaleSet = false;
        let isSeamScaleSet = false;
        this.resize(this.features, numImages); 
        this.resize(this.seamEstImgs, numImages); 
        this.resize(this.fullImgSizes, numImages); 

        let featureFindImgs = []; 
        let featureFindMasks = []; 
        for (let i = 0; i < numImages; i++) {
            featureFindImgs.push(this.genMatrix(height, width));
            featureFindMasks.push(this.genMatrix(height, width));

            if (this.regRes < 0) {
                featureFindImgs[i] = this.imgs[i];
                this.workScale = 1;
                isWorkScaleSet = true;
            } else {
                if (!isWorkScaleSet) {
                    this.workScale = Math.min(1.0, Math.sqrt(this.regRes * 1e6 / (this.fullImgSizes[i].height * this.fullImgSizes[i].width)));
                    isWorkScaleSet = true;
                }
                this.resize(this.imgs[i], featureFindImgs[i], this.workScale, this.workScale, INTER_LINEAR_EXACT);
            }
            if (!isSeamScaleSet) {
                this.seamScale = Math.min(1.0, Math.sqrt(this.seamEstRes * 1e6 / (this.fullImgSizes[i].height * this.fullImgSizes[i].width)));
                this.seamworkAspect = this.seamScale / this.workScale;
                isWorkScaleSet = true;
            }
            
            if (this.masks) {
                this.resize(this.masks[i], featureFindMasks[i], this.workScale, this.workScale, INTER_LINEAR_EXACT);
            }
        }

        this.computeImgageFeatures(this.featuresFinder, featureFindImgs, this.features, featureFindMasks);
        this.clear(featureFindImgs);
        this.clear(featureFindMasks);

        this.featuresMatcher(this.features, this.pairMathces, this.matchingMask);
        this.featuresMatcher.collectGarbage();

        let detail = new Detail();
        this.indices = detail.leaveBiggestComponet(this.features, this.pairMatches, this.confThress);
        let subImgs = [];
        let subSeamEstImgs = [];
        let subFullImgSizes = [];
        for (let i = 0; i < this.indices.length; i++) {
            subImgs.push(this.imgs[this.indices[i]]);
            subSeamEstImgs.push(this.seamEstImgs[this.indices[i]]);
            subFullImgSizes.push(this.fullImgSizes[this.indices[i]]);
        }
        this.seamEstImgs = subSeamEstImgs;
        this.imgs = subImgs;
        this.fullImgSizes = subFullImgSizes;

        if (this.imgs.length < 2) {
            return ERR_NEED_MORE_IMGS;
        }
        return OK;
    }

    estimateCameraParams() {
        if (!this.estimator(this.features, this.pairMatches, this.cameras)) {
            return ERR_HOMOGRAPHY_EST_FAIL;
        }
        this.bundleAdjuster.setConfThresh(this.confThresh);
        if (!this.bundleAdjuster(this.features, this.pairMatches, this.cameras)) {
            return ERR_CAMERA_PARAMS_ADJUST_FAIL;
        }
        let focals = [];
        for (let i = 0; i < this.cameras.length; i++) {
            focals.push(this.cameras[i].focal);
        }
        focals.sort((a, b) => a - b);
        if (focals.length % 2 === 1) {
            this.warpedImageScale = focals[Math.floor(focals.length / 2)];
        } else {
            this.warpedImageScale = (focals[Math.floor(focals.length / 2 - 1)] + focals[Math.floor(focals.length / 2)]) * 0.5;
        }

        if (this.doWaveCorrect) {
            let rMats = [];
            for (let i = 0; i < this.cameras.length; i++) {
                rMats.push(this.cameras[i].R);
            }
            let detail = new Detail();
            detail.waveCorrect(rMats, this.waveCorrectKind);
            for (let i = 0; i < this.cameras.length; i++) {
                this.cameras[i].R = rMats[i];
            }
        }

        return OK;
    }
}

