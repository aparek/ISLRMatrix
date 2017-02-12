%% Demo of (ISLR) estimating a low-rank and sparse matrix from noisy observation
% 
% Please cite as: 
% Improved Sparse and Low-Rank Matrix Estimation. (PrePrint)
% A. Parekh and I. W. Selesnick. Preprint https://arxiv.org/abs/1605.00042 
%
% Contact: Ankit Parekh, ankit.parekh@nyu.edu
% Last Edit: 11/24/16.
%% Initialize Definitions
clear, clc; close all;
rse = @(org, est) norm(org-est) / norm(org);
dB = @(x) 20 * log10(abs(x));
SNR = @(x,y) 10 * log10(sum(abs(x).^2)/sum(abs(x-y).^2));

%% Load test signal 

load TestSignal;
fs = 22050;
N = length(s); 
n = 0:N-1;

% Create noisy signal 
rng('default')
sigma = 0.03;
y = s + sigma*randn(size(s));

% Make transforms
R = 64; M = 2; K = 1; Nfft = 512;
[AH, A, normA] = MakeTransforms('STFT',N,[R M K Nfft]);

% Plot noise free data
figure(1), clf
subplot(2,1,1), plot(n/fs, s,'k')
box off
title('Noise free speech signal (y)')
xlim([0 N]/fs)
ylim([-0.3 0.3])

As = A(s);
subplot(2,1,2)
tt = R/M * ( (0:size(As, 2)) - 1 )/fs;    % tt : time axis for STFT
imagesc(tt, [0 0.5], dB(As(1:Nfft/2+1, :)), max(dB(As(:))) + [-50 -5])
axis xy
xlim([0 N]/fs)
ylim([0 0.3])
title('STFT of Noise-free data')
ylabel('Frequency')
xlabel('Time(s)')
colorbar

% Plot the noisy data
figure(2), clf
subplot(2,1,1), plot(n/fs, y,'k')

ylim([-0.3 0.3])
xlim([0 N]/fs)
box off
title(sprintf('Noisy speech signal (y), SNR = %2.2f dB', SNR(s,y)))

Ay = A(y);
subplot(2,1,2)
tt = R/M * ( (0:size(Ay, 2)) - 1 )/fs;  
imagesc(tt, [0 0.5], dB(Ay(1:Nfft/2+1, :)), max(dB(As(:))) + [-50 -5])
axis xy
colorbar
xlim([0 N]/fs)
ylim([0 0.3])
title('STFT of Noisy data')
ylabel('Frequency')
xlabel('Time(s)')

%% Estimate matrix X using atan and l1 penalty

lam1 = 0.029;
lam2 = 0.015;
mu = 1.5;
Nit = 20;
pen = 'atan';
[Ax, cost] = lrs_single(Ay,0.1,lam1,lam2,mu,pen,Nit); 
[AxL1,costL1] = lrs_single(Ay,0.1,0.025,0.009,mu,'l1',Nit); 


%% Plot cost function history

figure(3), clf
plot(cost, 'k'); hold on
plot(costL1, ':.k')
legend('ISLR', 'SLR')
title('Cost function history for atan (ISLR) and L1 (SLR)')
box off

%%
figure(4), clf

subplot(4,1,1)
tt = R/M * ( (0:size(As, 2)) - 1 )/fs;    % tt : time axis for STFT
imagesc(tt, [0 1], dB(As(1:Nfft/2+1, :)), [-65 -20])
axis xy
title('(a) Clean speech spectrogram')
xlim([0 N]/fs)
ylim([0 0.3])
colorbar
ylabel('Frequency (kHz)')


subplot(4,1,2)
tt = R/M * ( (0:size(Ay, 2)) - 1 )/fs;    % tt : time axis for STFT
imagesc(tt, [0 1], dB(Ay(1:Nfft/2+1, :)),[-65 -20])
axis xy
title(sprintf('(b) Noisy speech spectrogram. SNR = %2.2f dB', SNR(s,real(AH(Ay)))))
xlim([0 N]/fs)
ylim([0 0.3])
ylabel('Frequency (kHz)')
colorbar

subplot(4,1,3)
tt = R/M * ( (0:size(AxL1, 2)) - 1 )/fs;    % tt : time axis for STFT
imagesc(tt, [0 1], dB(A(AH(AxL1(1:Nfft/2+1, :)))),[-65 -20])
axis xy
title(sprintf('(c) SLR estimate. SNR = %2.2f dB', SNR(s,real(AH(AxL1)))))
xlim([0 N]/fs)
ylim([0 0.3])
ylabel('Frequency (kHz)')
colorbar

subplot(4,1,4)
tt = R/M * ( (0:size(Ax, 2)) - 1 )/fs;    % tt : time axis for STFT
imagesc(tt, [0 1], dB(A(AH(Ax(1:Nfft/2+1, :)))),[-65 -20])
axis xy
title(sprintf('(d) ISLR (proposed) estimate. SNR = %2.2f dB', SNR(s,real(AH(Ax)))))
set(gca,'XTick',0:0.05:0.2,'YTick',0:0.1:0.3)
xlim([0 N]/fs)
ylim([0 0.3])
ylabel('Frequency (kHz)')
box off
xlabel('Time (s)')
colorbar

%% Comparison of SVD values

figure(5), clf
plot(svd(Ay),'color',[0.5 0.5 0.5],'Marker','.'); hold on
plot(svd(As),'.-k');
plot(svd(Ax),'.--k');
plot(svd(AxL1),'.:k');
box off
title('Comparison of Singular Values')
xlabel('index (k)')
ylabel('Singular value \sigma_k(X)')
legend('Noisy','Clean','ISLR','SLR')
xlim([1 12])