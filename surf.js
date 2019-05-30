import Jimp from 'jimp';

const ORI_RADIUS = 6;
const ORI_WIN = 60;
const PATCH_SZ = 20;

class SURF {
    constructor() {
        this.upright = 0;
        this.sampleStep = 2;
        this.sum = [];
        this.maskSum = [];
        this.sizes = [];
        this.dets = [];
        this.traces = [];
        this.octaves = [];
        this.layers = [];
        this.extended = false;
    }

    genZeros(height, width) {
        let arr = [];
        let res = [];
        for (let i = 0; i < height; i++) {
            arr.push(0);
        }
        for (let i = 0; i < width; i++) {
            res.push(arr);
        }
        return res;
    }

    calcHaarPattern(src, hf, N, hOffset, offset) {
        let d = 0;
        const width = src[0].length;
        const { hOffset, wOffset } = offset;
        if (width) {
            for (let k = 0; k < N; k++) {
                // d += (src[hf[k][0]] + src[hf[k][3]] - src[hf[k][1]] - src[hf[k][2]]) * hf[k][4];
                // TODO wOffset
                d += (src[hOffset + Math.floor(hf[k][0] / width)][(wOffset + hf[k][0]) % width] + 
                    src[hOffset + Math.floor(hf[k][3] / width)][(wOffset + hf[k][3]) % width] - 
                    src[hOffset + Math.floor(hf[k][1] / width)][(wOffset + hf[k][1]) % width] - 
                    src[hOffset + Math.floor(hf[k][2] / width)][(wOffset + hf[k][2]) % width]) * hf[k][4]; 
            }
        }
        return d;
    }

    // dst === idx
    resizeHaarPattern(src, dst, oldSize, newSize, width) {
        const ratio = newSize / oldSize;
        for (let k = 0; k < dst.length; k++) {
            const dx1 = Math.round(ratio * src[k][0]);
            const dy1 = Math.round(ratio * src[k][1]);
            const dx2 = Math.round(ratio * src[k][2]);
            const dy2 = Math.round(ratio * src[k][3]);
            dst[k][0] = dy1 * width + dx1;
            dst[k][1] = dy2 * width + dx1;
            dst[k][2] = dy1 * width + dx2;
            dst[k][3] = dy2 * width + dx2;
            // weight
            dst[k][4] = src[k][4] / ((dx2 - dx1) * (dy2 - dy1));
        }
    }

    calcLayerDetAndTrace(sum, size, sampleStep, det, trace) {
        if (size > sum.length - 1 || size > sum[0].length - 1) {
            return;
        }

        const NX = 3;
        const NY = 3;
        const NXY = 4;
        const dx_s = [[0, 2, 3, 7, 1], [3, 2, 6, 7, -2], [6, 2, 9, 7, 1]];
        const dy_s = [[2, 0, 7, 3, 1], [2, 3, 7, 6, -2], [2, 6, 7, 9, 1]];
        const dxy_s = [[1, 1, 4, 4, 1], [5, 1, 8, 4, -1], [1, 5, 4, 8, -1], [5, 5, 8, 8, 1]];
        const height = sum.length;
        const width = sum[0].length;

        const Dx = this.genZeros(NX, 5); 
        const Dy = this.genZeros(NY, 5); 
        const Dxy = this.genZeros(NXY, 5); 

        this.resizeHaarPattern(dx_s, Dx, 9, size, width);
        this.resizeHaarPattern(dy_s, Dx, 9, size, width);
        this.resizeHaarPattern(dxy_s, Dxy, 9, size, width);

        const heightRange = 1 + (height - 1 - size) / sampleStep;
        const widthRange = 1 + (width - 1 - size) / sampleStep;
        const margin = Math.round(0.5 * size / sampleStep);
        // TODO
        for (let i = 0; i < heightRange; i++) {
            let sumIdx = i * sampleStep;
            let sumOffset = {
                hOffset: Math.floor(sumIdx / width),
                wOffset: sumIdx % width
            };
            for (let j = 0; j < widthRange; j++) {
                const dx = this.calcHaarPattern(sum, Dx, 3, sumOffset);
                const dy = this.calcHaarPattern(sum, Dy, 3, sumOffset);
                const dxy = this.calcHaarPattern(sum, Dxy, 4, sumOffset);
                sumIdx += sampleStep;
                sumOffset = {
                    hOffset: Math.floor(sumIdx / width),
                    wOffset: sumIdx % width
                };
                det[i + margin][j + margin] = dx * dy - 0.81 * dxy * dxy;
                trace[i + margin][j + margin] = dx + dy;
            }
        }
    }

    interpolateKeyPoint(N9, dx, dy, ds, kpt) {
        const b = [-(N9[1][5] - N9[1][3]) * 0.5 , -(N9[1][7] - N9[1][1]) * 0.5, -(N9[2][4] - N9[0][4]) * 0.5];
        const A = [N9[1][3] - 2 * N9[1][4] + N9[1][5],
            (N9[1][8] - N9[1][6] - N9[1][2] + N9[1][0]) * 0.25,
            (N9[2][5] - N9[2][3] - N9[0][5] + N9[0][3]) * 0.25,
            (N9[1][8] - N9[1][6] - N9[1][2] + N9[1][0]) * 0.25,
            N9[1][1] - 2 * N9[1][4] + N9[1][7],
            (N9[2][7] - N9[2][1] - N9[0][7] + N9[0][1]) * 0.25,
            (N9[2][5] - N9[2][3] - N9[0][5] + N9[0][3]) * 0.25,
            (N9[2][7] - N9[2][1] - N9[0][7] + N9[0][1]) * 0.25,
            N9[0][4] - 2 * N9[1][4] + N9[1][4]];

        // TODO
        const x = A.solve(b, decomp_lu);
        const ok = (x[0] !== 0 || x[1] !== 0 || x2 !== 0) &&
            Math.abs(x[0]) <= 1 && Math.abs(x[1]) <= 1 && Math.abs(x[2]) <= 1;

        if (ok) {
            kpt.pt.x += x[0] * dx;
            kpt.pt.y += x[1] * dy;
            kpt.size = Math.round(kpt.size + x[2] * ds);
        }
        return ok;
    }

    findInvoker(sum, mask_sum, dets, traces, sizes, sampleSteps, middleIndices, keypoints, octaveLayers, hessianThreshold) {

    }

    findMaxInLayer(sum, mask_sum, dets, traces, sizes, keypoints, octave, layer, hessianThreshold, sampleStep) {
        const dm = [0, 0, 9, 9, 1];
        const Dm = this.genZeros(1, 5);
        const size = sizes[layer];
        const layer_rows = (sum.length - 1) / sampleStep;
        const layer_cols = (sum[0].length - 1) / sampleStep;
        const margin = (sizes[layer + 1] / 2) / sampleStep + 1;
        if (!mask_sum) {
            this.resizeHaarPattern(dm, Dm, 9, size, mask_sum[0].length);
        }
        // TODO elemSize
        const step = Math.floor(dets[layer].step / dets[layer].elemSize());
        for (let i = margin; i < layer_rows - margin; i++) {
            for (let j = margin; j < layer_cols - margin; j++) {
                const val = dets[layer][j];
                if (val > hessianThreshold) {
                    const sum_i = sampleStep * (i - size * 0.5 / sampleStep);
                    const sum_j = sampleStep * (j - size * 0.5 / sampleStep);
                    const N9 = [[dets[layer - 1][-step - 1], dets[layer - 1][-step], dets[layer - 1][-step + 1],
                        dets[layer - 1][-1], dets[layer - 1][0], dets[layer - 1][1],
                        dets[layer - 1][step - 1], dets[layer - 1][step], dets[layer - 1][step + 1]],
                        [dets[layer1][-step - 1], dets[layer1][-step], dets[layer][-step + 1],
                            dets[layer][-1], dets[layer][0], dets[layer][1],
                            dets[layer][step - 1], dets[layer][step], dets[layer][step + 1]],
                        [dets[layer + 1][-step - 1], dets[layer + 1][-step], dets[layer + 1][-step + 1],
                            dets[layer + 1][-1], dets[layer + 1][0], dets[layer + 1][1],
                            dets[layer + 1][step - 1], dets[layer + 1][step], dets[layer + 1][step + 1]]];

                    if (!mask_sum) {
                        const mval = this.calcHaarPattern(mask_sum, Dm, 1);
                        if (mval < 0.5) {
                            continue;
                        }
                    }

                    // TODO N9 max
                    if(val > Math.max(N9)) {
                        const center_i = sum_i + (size - 1) * 0.5;
                        const center_j = sum_j + (size - 1) * 0.5;

                        kpt(center_j, center_i, sizes[layer] - 1, -1, val, octave, (traces[layer][j] > 0) - (traces[layer][j] < 0));
                        const ds = size - sizes[layer - 1];
                        const interp_ok = this.interpolageKeypoint(N9, sampleStep, sampleStep, ds, kpt);

                        if (interp_ok) {
                            // TODO
                            // cv::AutoLock
                            keypoints.push(kpt);
                        }
                    }
                }
            }
        }
    }

    genKeyPoint() {
    }

    fastHessianDetector(sum, mask_sum, keypoints, octaves, octaveLayers, hessianThreshold) {
        const sampleStep0 = 1;
        const totalLayers = (octaveLayers + 2) * octaves;
        const middleLayers = octaveLayers * octaves;
        // TODO
        const dets = [];
        const traces = [];
        const size = [];
        const sampleSteps = [];
        const middleIndices = [];

        // clear()
        keypoints.splice(0, keypoint.length);
        let index = 0;
        let middleIndex = 0;
        let step = sampleStep0;
        for (let octave = 0; octave < octaves; octave++) {
            for (let layer = 0; layer< octaveLayers; layer++) {
                // TODO
                detx[index].push();
                sampleSteps[index] = step;
                if (0 < layer && layer <= octaveLayers) {
                    middleIndices[middleIndex] = index;
                    middleIndex += 1;
                }
                index += 1;
            }
            step *= 2;
        }
        findInvoker(sum, mask_sum, dets, traces, sizes, sampleSteps, middleIndices, keypoints, octaveLayers, hessianThreshold);
        // TODO
        keypoints.sort();
    }

    invoker() {
        const oriSampleBound = (2 * ORI_RADIUS + 1) * (2 * ORI_RADIUS + 1);
        // TODO
        const apt = [];
        const aptw = [];
        const DW = p[];
        // TODO
        const gOri = getGaussianKernel();
        const oriSample = 0;
        // TODO
        // for () {...
        const gDest = getGaussianKernel();
        for (let i = 0; i < PATCH_SZ; i++) {
            for (let j = 0; j < PATCH_SZ; j++) {
                DW[i * PATCH_SZ + j] = gDest[i][0] * gDest[j][0];
            }
        }
    }

    invokerOperator() {
        const NX = 2;
        const NY = 2;
        const dx_s =[[0, 0, 2, 4, -1], [2, 0, 4, 4, 1]];
        const dy_s =[[0, 0, 4, 2, 1], [0, 2, 4, 4, -1]];
        const oriSampleBound = (2 * ORI_RADIUS + 1) * (2 * ORI_RADIUS + 1);
        const X = [];
        const Y = [];
        const angle = [];
        const patch = this.genZeros(PATCH_SZ + 1, PATCH_SZ + 1);
        const _patch = this.genZeros(PATCH_SZ + 1, PATCH_SZ + 1);
        const DX = [];
        const DY = [];
        const dsize = this.extended ? 128: 64;
        // TODO
        const start = 0;
        const end = 0;
        const height = sum.length;
        const width = sum[0].length;

        // TODO
        // seems no need
        // let maxSize = 0;
        // for (let k = start; k < end; k++) {
        //     maxSize = Math.max(maxSize, keypoints[k].length);
        // }
        // const imaxSize = Math.max(1, Math.ceil((PATCH_SZ + 1) * maxSize * 1.2 / 9.0));
        // winbuf

        for (let k = k1; k < k2; k++) {
            let vec = [];
            let dx_t = [];
            let dy_t = [];
            let kp = {};
            let size = kp.length;
            let s = size * 1.2 / 9.0;
            const center = kp.pt;
            let gradWaveSize = 2 * Math.round(2 * s);
            if (height < gradWaveSize || width < gradWaveSize) {
                kp.splice(0, kp.length); 
                continue;
            }
            let descriptorDir = 270.0;
            if (this.upright === 0) {
                this.resizeHaarPttern(dx_s, dx_t, NX, 4, gradWaveSize, sum[0].length); 
                this.resizeHaarPttern(dy_s, dy_t, NX, 4, gradWaveSize, sum[0].length); 
                for (let k1 = 0, let nangle = 0; k1 < oriSamples; k1++) {
                    const x = Math.round(center.x + apt[k1].x * s - (gradWaveSize - 1) * 0.5);
                    const y = Math.round(center.y + apt[k1].y * s - (gradWaveSize - 1) * 0.5);
                    if (y < 0 || y >= height - gradWaveSize ||
                        x < 0 || x >= width - gradWaveSize) {
                        continue;
                    }
                    const sumOffset = {
                        hOffset: y,
                        wOffset: x 
                    };
                    const vx = this.calcHaarPattern(sum, dx_t, 2, sumOffset);
                    const vy = this.calcHaarPattern(sum, dy_t, 2, sumOffset);
                    X[nangle] = vx * aptw[k1];
                    Y[nangle] = vy * aptw[k1];
                    nangle += 1;
                }
                if (nangle === 0){
                    kp.splice(0, kp.length); 
                    continue;
                }
                //TODO phase
                phase(X, Y, angle, true);
                let bestX = 0;
                let bestY = 0;
                let descriptorMod = 0;
                for (let i = 0; i < 360; i++) {
                    let sumX = 0;
                    let sumY = 0;
                    let tempMod = 0;
                    for (let j = 0; j < nangle; j++) {
                        let d = Math.abs(Math.round(angle[j]) - i);
                        if (d < ORI_WIN / 2 || d < 360 - ORI_WIN / 2) {
                            sumX += X[j];
                            sumY += Y[j];
                        }
                    }
                    tempMod = sumX * sumX + sumY * sumY;
                    if (tempMod > descriptorMod) {
                        descriptorMod = tempMod;
                        bestX = sumX;
                        bestY = sumY;
                    }
                }
                descriptorDir = Math.atan2(-bestY, bestX);
            }
            kp.angle = descriptorDir;
            if (!descriptor || !descriptor.data) {
                continue;
            }

            const winSize = Math.round((PATCH_SZ + 1) * s);
            const win = this.genZeros(winSize, winSize);
            if (!upright) {
                descriptorDir *= Math.PI / 180.0;
                const sinDir = - Math.sin(descriptorDir);
                const cosDir = Math.cos(descriptorDir);
                const winOffset = -(win_size - 1) * 0.:
                const startX = center.x + winOffset * cosDir + winOffset * sinDir;
                const starty = center.x - winOffset * cosDir + winOffset * sinDir;
                const width1 = img.bitmap.width - 1;
                const height1 = img.bitmap.height - 1;
                // TODO step
                const imgStep = img.step;
                const data = img.bitmap.data;
                for (let i = 0; i < winSize; i++, startX += sinDir, startY += cosDir) {
                    const pixelX = startX;
                    const pixelY = startY;
                    for (let j = 0; j < winSize; j++, pixelX += cosDir, pixelY -= sinDir) {
                        const ix = Math.floor(pixelX);
                        const iy = Math.floor(pixelY);
                        if (ix < width1 && iy < height1) {
                            const a = pixelX - ix;
                            const b = pixelY - iy;
                            win[i][j] = Math.round(data[0] * (1.0 - a) * (1.0 - b) + data[1] * a * (1.0 - b) + 
                                data[imgStep] * (1.0 - a) * b + data[imgStep + 1] * a * (1.0 - b));
                        } else {
                            const x = Math.min(Math.max(Math.round(pixelX), 0), width1);
                            const y = Math.min(Math.max(Math.round(pixelY), 0), width1);
                            win[i][j] = data[y][x];
                        }
                    }
                } else {
                    const winOffset = -(winSize - 1) * 0.5;
                    const startX = Math.round(center.x + winOffset);
                    const startY = Math.round(center.y - winOffset);
                    for (let i = 0; i < winSize; i++, startX++) {
                        const pixelX = startX;
                        const pixelY = startY;
                        for (let j = 0; j < winSize; j++, pixelY--) {
                            const x = Math.max(pixelX, 0);
                            const y = Math.max(pixelY, 0);
                            x = Math.min(x, width1);
                            y = Math.min(y, height1);
                            win[i][j] =  data[y][x];
                        }
                    }
                }
                // TODO resize
                resize(win, patch, patch.length, 0, 0, INTER_AREA);
                for (let i =  0; i < PATCH_SZ; i++) {
                    for (let j =  0; j < PATCH_SZ; j++) {
                        const dw = DW[i * PATCH_SZ + j];
                        cosnt vx = (PATCH[i][j + 1] - PATCH[i][j] + PATCH[i + 1][j + 1]  PATCH[i + 1][j]) * dw; 
                        cosnt vx = (PATCH[i + 1][j] - PATCH[i][j] + PATCH[i + 1][j + 1]  PATCH[i][j + 1]) * dw; 
                        DX[i][j] = vx;
                        DX[i][j] = vy;
                    }
                    vec = descriptors[k];
                    for (let k1 = 0; k1 < dsize; k1++) {
                        vec[k1] = 0;
                    }
                    let squareMag = 0;
                    if (this.extended) {
                        for (let i = 0; i < 4; i++) {
                            for (let j = 0; j < 4; j++) {
                                for (let y = i * 5; y < i * 5 + 5; y++) {
                                    for (let x = j * 5; x < j * 5 + 5; x++) {
                                        const tx = DX[y][x];
                                        const ty = DY[y][x];
                                        // TODO fabs
                                        if (ty >= 0) {
                                            vec[0] += tx;
                                            vec[1] += fabs(tx);
                                        } else {
                                            vec[2] += tx;
                                            vec[3] += fabs(tx);
                                        }
                                        if (tx >= 0) {
                                            vec[4] += ty;
                                            vec[5] += fabs(ty);
                                        } else {
                                            vec[6] += ty;
                                            vec[7] += fabs(ty);
                                        }
                                    }
                                }
                                for (let k1 = 0; k1 < 8; k1++) {
                                    squareMag += vec[k1] * vec[k1];
                                }
                                // TODO
                                vec += 8;
                            }
                        }
                    } else {
                        for (let i = 0; i < 4; i++) {
                            for (let j = 0; j < 4; j++) {
                                for (let y = i * 5; y < i * 5 + 5; y++) {
                                    for (let x = j * 5; x < j * 5 + 5; x++) {
                                        const tx = DX[y][x];
                                        const ty = DY[y][x];
                                        vec[0] += tx;
                                        vec[1] += ty;
                                        vec[2] += fabs(tx);
                                        vec[3] += fabs(ty);
                                    }
                                }
                                for (let k1 = 0; k1 < 4; k1++) {
                                    squareMag += vec[k1] * vec[k1];
                                }
                                // TODO
                                vec += 4;
                            }
                        }
                    }
                    vec = descriptors[k];
                    const scale = 1.0 / (Math.sqrt(squareMag + 1e-6));
                    for (let k1 = 0; k1 < 4; k1++) {
                        vec[k1] *= scale;
                    }
                }
            }
        }
    }

    detectAndCompute(img, mask, keypoints, descriptors, useProvidedKeypoints) {
        img.greyscale();
        // TODO
        const sum = integral(img);
        if (!useProvidedKeyPoints) {
            if (!mask) {
                const mask1 = Math.min(mask, 1);
                const maskSum = integral(mask1);
            }
            fastHessianDetector(sum, maskSum, keypoints, octaves, octaveLayers, hessianThreshold);
            if (!mask) {
                for (let i = 0; i < keypoints.length;) {
                    const pt = keypoints[i].pt;
                    if (mask(pt.y, pt.x) === 0) {
                        // TODO
                        keypoints.erase(i);
                        continue; // keep "i"
                    }
                    i++;
                }
            }
        }
        const N = keypoints.length();
        if (doDescriptors) {
            // const is1D = descriptors.type === '1D';
            // if (is1D) {
            // TODO
            // } else {
            // }
            createDescriptor();
        }
        this.invoker();
        for (let i = j = 0; i < N; i++) {
            if (kepoints[i].length > 0 {
                if (i > j) {
                    keypoints[j] = keypoints[i];
                    if (doDescriptors) {
                        // TODO
                        // memcpy(descriptors.ptr(j), descriptors.ptr(i), dsize);
                    }
                }
                j += 1;
            }
        }
        if (N > j) {
            // TODO
            keypoints.resize(N);
            if (doDescriptors) {
                const d = [];
                for (let k = 0; k < N; k++) {
                    d.push(descriptors[k]);
                }
                // this.descriptors = d;

            }
        }
    }

    create(threshold, octaves, layers, extended, upright) {
    }

}
