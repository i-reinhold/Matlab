function [MTRS, MTS, f, t] = mtreassignspec(x, K, fs, p, p_option, nfft, tfact, e)
% Calculates the multitaper reassigned spectrogram and multitaper
% spectrogram, designed for transient oscillating signals with Gaussian
% envelopes. Uses overlapping Hermite windows. The multitaper reassigned
% spectrogram will be a matrix of dim nfft/2 x ceil(signal length/t_fact).
% Implemented according to method by Reinhold and Sandsten (2022)
% doi: 10.1016/j.sigpro.2022.108570
%
% Output
% MTRS      Multitaper reassigned spectrogram.
% MTS       Multitaper spectrogram.
% f         Frequency vector.
% t         Time vector.
%
% Input
% x         Signal.
% K         Number of windows.
% fs        Sampling frequency.
% p         Window scaling or length parameter, see p_option.
% p_option  Option for p, defined as either: "sigma" - scaling paramater of
%           Gaussian function, "fwhm" - length according to full width at
%           half maximum, "p96" - length according to 96% of energy is full
%           width, default p_option="sigma".
% nfft      Number of frequency points evaluated in fft, default nfft=1024.
% tfact     Downscaling factor for time axis, default t_fact=1, i.e. no
%           downscaling.
% e         Energy threshold (less than 1), default e = 1e-4.
%
% Notes
% Set K = 1 for the scaled reassigned spectrogram, Sandsten and Brynolfsson
% (2015) doi: 10.1109/LSP.2014.2350030
%
% Implemented by: Isabella Reinhold, Lund University

if nargin < 5 || isempty(p_option)
    p_option = "sigma";
end

if nargin < 6 || isempty(nfft)
    nfft = 1024;
end

if nargin < 7  || isempty(tfact)
    tfact = 1;
end

if nargin < 8  || isempty(e)
    e = 10^(-4);
end

% Determine window scaling from length
if p_option == "fwhm"
    p = p / (2* sqrt(2 * log(2)));
elseif p_option == "p96"
    p = p / 4;
end
% Convert scaling to samples
p = p * fs;

% Hermite window
N = length(x);
[Win, tWin, dWin] = hermitewin(K, p, N);

% Spectrogram and STFTs
[S, F, tF, dF] = stftgen(x, Win, tWin, dWin, nfft, tfact);

% Reassignment
[MTRS, MTS] = screassign(S, F, tF, dF, p, tfact, e);

% Even/odd
if rem(N, 2) ~= 0
    MTS = MTS(:, 1:N);
    MTRS = MTRS(:, 1:N);
end

% Frequency vector
f = (0:nfft/2-1) * fs / nfft;

% Time vector
t = (0:ceil(N/tfact)-1) / fs * tfact;
end

% ---------- HELP FUNCTIONS ----------

function [Win, tWin, dWin] = hermitewin(K, p, N)

if nargin < 3
    Error('Error in input for hermitewin')
end

% Time vector
M = min(12*p, N);
tvect = (-M/2:M/2-1)';
M = length(tvect);

% Polynomials (physicists')
He = zeros(M, K);
He(:, 1) = ones(M, 1);
if K > 1
    He(:, 2) = 2*tvect/p;
    for k = 3:K
        He(:, k) = 2*tvect/p.*He(:, k-1) - 2*(k-2)*He(:, k-2);
    end
end

% Unit energy weight function
wfun = exp(-(tvect.^2)/(2*p^2));
wfun = wfun / sqrt(sqrt(pi)*p);

% K:th order window
Win = He .* wfun;

% Time multiplied window
tWin = tvect .* Win;

% Differentiated window
dWin = zeros(size(Win));
dWin(:, 1) = - tvect/p^2.*He(:, 1) .* wfun;
for k = 2:K
    dWin(:, k) = 2*(k-1)/p*He(:, k-1) .* wfun - tvect/p^2.*He(:, k) .* wfun;
end

% Unit energy (polynomial)
for k = 1:K
    E = norm(Win(:, k));
    tWin(:, k) = tWin(:, k) / E;
    dWin(:, k) = dWin(:, k) / E;
    Win(:, k) = Win(:, k) / E;
end
end

function [S, F, tF, dF] = stftgen(x, Win, tWin, dWin, nfft, tfact)

if nargin < 6
    error('Error in input for stftgen');
end

% Window length and number of windows
[M, K] = size(Win);

% Zeropad signal
if rem(length(x),2) ~= 0
    x = [zeros(fix(M/2), 1) ; x ; zeros(fix(M/2+1), 1)];
else
    x = [zeros(fix(M/2), 1) ; x ; zeros(fix(M/2), 1)];
end
N = length(x);

% STFTs (positive frequencies)
E = sqrt(nfft / tfact);
F = zeros(nfft/2, ceil((N-M)/tfact), K);
tF = zeros(nfft/2, ceil((N-M)/tfact), K);
dF = zeros(nfft/2, ceil((N-M)/tfact), K);
ind = 0;
for j = 1:tfact:N-M
    ind = ind + 1;
    step_x = x(j:j-1+M);
    step_F = fft(Win.*step_x, nfft) / E;
    step_tF = fft(tWin.*step_x, nfft) / E;
    step_dF = fft(dWin.*step_x, nfft) / E;
    F(:, ind, :) = step_F(1:nfft/2, :);
    tF(:, ind, :) = step_tF(1:nfft/2, :);
    dF(:, ind, :) = step_dF(1:nfft/2, :);
end

% All spectrograms
S = abs(F).^2;
end

function [MTRS, MTS] = screassign(S, F, tF, dF, p, tfact, e)

if nargin < 7
    error('Error in input for screassign')
end

% Number of frequency bins, time bins, and windows
[M, N, ~] = size(S);

% K reassignment vectors
[mesht, meshf] = meshgrid(1:N, 1:M);
tmat0 = 1 / tfact * real((tF - p^2*dF) ./ F);
fmat0 = 1 * M/pi * imag((dF - tF/p^2) ./ F);
tmat = mesht + tmat0;
fmat = meshf - fmat0;

% Energy threshold and within bounds reassignment
Se = e * max(S, [], [1, 2]);
Scheck = S>Se & fmat>=1 & fmat<=M & tmat>=1 & tmat<=N;

% Averaging reassignment vectors
mtmat = round(sum(Scheck .* tmat,3) ./ sum(Scheck,3));
mfmat = round(sum(Scheck .* fmat,3) ./ sum(Scheck,3));

% Multitaper spectrogram and new checking matrises
MTS = mean(S,3);

% New energy threshold
MTSe = e * max(MTS, [], [1, 2]);

% Reassignment
MTRS = zeros(M,N);
for n = 1:N
    for m = 1:M
        if MTS(m, n)>MTSe && mfmat(m, n)>0 && mfmat(m, n)<=M && mtmat(m, n)>0 && mtmat(m, n)<=N
            new_f = mfmat(m,n);
            new_t = mtmat(m,n);
            MTRS(new_f, new_t) = MTRS(new_f, new_t) + MTS(m,n);
        end
    end
end
end
