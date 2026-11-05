classdef SkillGeneralisation
    % This SkillGeneralisation class implements the Dynamic Movement Primitives (DMPs).
    % It includes methods for preprocessing demonstration trajectories, performing nonlinear fitting,
    % and reconstructing trajectories. Some functions in this class are incomplete
    % and need to be implemented based on the knowledge from the Robot Learning and Teleoperation module.
    properties
        % Parameters for the DMPs
        numStates double            % Number of activation functions (i.e., number of RBF in the forcing term or K of a GMM )
        numVar int8                 % Number of variables dim([x,s1,s2]) (decay term and perturbing force)
        numVarPos int8              % Dimension of spatial variables dim([s1,s2]) 
        numData int16               % Number of time stamps of a trajectory 
        numDemos int16              % Number of demonstrations 
        beta double                 % Stiffness gain (β)
        alpha double                % Damping gain (α)
        dt double                   % Duration of time step (τ)
        L double                    % Feedback term
        xIn = []                    % State of the canonical system
        dataTraining = []           % Forcing term for training
        startPos double             % Start position of the trajectory
        endPos double               % End position of the trajectory
    end

    methods
        function obj = SkillGeneralisation(dim, states, beta, alpha, dt)
            % Construct an instance of this class
            obj.numStates = states;                 % Number of activation functions 
            obj.numVar = dim;                       % Number of variables
            obj.numVarPos = obj.numVar-1;           % Dimension of spatial variables
            obj.beta = beta;                        % Stiffness gain
            obj.alpha = alpha;                      % Damping gain (with ideal underdamped damping ratio) 
            obj.dt = dt;                            % Duration of time step
            obj.L = [eye(obj.numVarPos)*beta, eye(obj.numVarPos)*alpha]; % Feedback term
        end

        %% Generate time Stamps
        function obj = canonicalSystemInitialisation(obj,decayFactor,numData)
            obj.xIn(1) = 1;
            for t = 2:numData
            	obj.xIn(t) = obj.xIn(t-1) - decayFactor * obj.xIn(t-1) * obj.dt;       % Update of decay term (dx/dt = -ax, τx'=-ax)
            end
           
            obj.numData = numData;
        end

        %% Compute imaginary force from trajectory
        function obj = trajectort2Forcing(obj, demos)
            obj.numDemos = length(demos);
            numTimeSteps = length(obj.xIn);
            obj.endPos = demos{1}.pos(:,end);
            
            obj.startPos = demos{1}.pos(:,1);
            for n = 1:obj.numDemos
                pos = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),numTimeSteps));     % Resampling Positions
                [traj(n).pos, traj(n).vel, traj(n).acc] = obj.generateHighOrderTerms(pos);
                oneDemoForceWithTimeStamps = [obj.xIn; (traj(n).acc - (repmat(obj.endPos,1,numTimeSteps)-traj(n).pos)*obj.beta + traj(n).vel*obj.alpha) ./ repmat(obj.xIn,obj.numVarPos,1)];
                obj.dataTraining = [obj.dataTraining, oneDemoForceWithTimeStamps];
                
            end
        end
        
        %% Compute velocity and acceleration from position trajectory 
        function [pos, vel, acc] = generateHighOrderTerms(obj, pos)
            vel = gradient(pos) / obj.dt;             % Compute Velocity
            acc = gradient(vel) / obj.dt;             % Compute Acceleration
        end

        %% Ordinary DMP with locally weighed locally WLS
        function fOut = fittingWithLocallyWLS(obj)
           
              x = obj.xIn;   % 1*200
              
              mu = linspace(x(1),x(end),obj.numStates);
            
    
              sigma = (mu(1) - mu(2))^2;
              
          
              x_full = repmat(obj.xIn,1,obj.numDemos);
              
              psi = exp(-0.5 * (x_full - mu').^2 / sigma);

             

              H = psi ./ (sum(psi, 1) + eps);           % 9 * 1000

              amp = obj.endPos - obj.startPos;

              aWeight = zeros(obj.numVarPos,obj.numStates);
              f_target = obj.dataTraining(2:end,:);  %2*1000
                for i = 1:obj.numStates
               
                    W = diag(H(i, :));
                
                    for d = 1:obj.numVarPos
                    
                        basis_X = (x_full * amp(d))'; 
                        target_Y = f_target(d, :)';
                    
                   
                    aWeight(d, i) = (basis_X' * W * basis_X) \ (basis_X' * W * target_Y);
                    end
                end
                
                psi_single = exp(-0.5 * (obj.xIn - mu').^2 / sigma);          % 9 x 200
                H_single   = psi_single ./ (sum(psi_single, 1) + eps);         % 9 x 200
                
                fOut = aWeight * H_single ...
                .* repmat(obj.xIn,      obj.numVarPos, 1) ...             % 2 x 200
                .* repmat(amp,    1,    obj.numData);             
            
                
            

          end

        %% Optimised DMP with GMM and GMR   
        function fOut = fittingWithGMR(obj)
            % -------------------Add your code here --------------------    👈👈👈
            gmmData = obj.dataTraining';   %(3*1000) x,f1,f2
            amp = obj.endPos - obj.startPos;

            for d = 1:obj.numVarPos
               gmmData(:, d+1) = gmmData(:, d+1) / amp(d);
            end
            K = obj.numStates; 
            gmmodel = MixtureGaussians(gmmData, K);
            gmmodel = gmmodel.gmmFit(gmmData,60, 1e-6, false);
            mask = [1 zeros(1, obj.numVarPos)];
            %mask = [1 0 0];
            gmmodel = gmmodel.defineQueryDim(mask);
            query = obj.xIn';   
            gmmodel = gmmodel.gaussianMixtureRegression(query);
            % GMM(R) Plots, debugging use
            if true
                %figure;     % Create a 3D plot of the GMM and the data points
                %gmmodel.plotGmmAndData(gmmData);
                figure;     % Create the log-likelihood curve during the EM iterations
                plot(gmmodel.logLikelihood,'-r','LineWidth',2);
                title('Log-likelihood during training');
                xlabel('number of iterations')
                ylabel('Log-likelihood')
                %figure;     % Display the demonstrations and the reproduced reference trajectory in one figure
                %scatter3(gmmData(:,1), gmmData(:,2), gmmData(:,3), '.b'); hold on;
                %scatter3(gmmodel.regressedTraj(:,1), gmmodel.regressedTraj(:,2), gmmodel.regressedTraj(:,3), '.g');
                hold off
            end
            figure;
            %gmmodel.plotGmrResults(gmmData);
           % amp = [-3;-3] - obj.startPos;        % to debug
            
            fOut = gmmodel.regressedTraj(:, 2:end)';   % return as (dim x time)   %200*2
            fOut = fOut .* repmat(obj.xIn, obj.numVarPos, 1).* repmat(amp,     1, obj.numData);

      
        end

        %% Generalise imaginary force back to trajectory
        function trajOut = forcing2Trajectory(obj, forcingTraj)
        
            x = obj.startPos;% Current position (starts at y0)
            
           

            xTarget = obj.endPos;         % Current goal (g)
           % xTarget = [-3;-3];
            dx = zeros(obj.numVarPos,1);  % Current velocity (starts at 0)
            rData = zeros(obj.numVarPos, obj.numData); % Pre-allocate for speed
            for t = 1:obj.numData
                Nudge = forcingTraj(:,t) ;                  
                ddx =    (obj.beta * (xTarget - x) - obj.alpha * dx) + Nudge;
                dx = dx + ddx * obj.dt; 
            	x = x + dx * obj.dt;
            	rData(:,t) = x;
            end
            trajOut = rData;
        end


        
 
    end




    

methods (Static)              
      
 


function compareKvalues(dataTraining, demos_in, xIn, nbData, kRange, beta, alpha, dt, endPos, startPos, numVarPos)

    % Compute mean demo as reference
    meanDemo = zeros(2, nbData);
    for n = 1:length(demos_in)
        pos      = spline(1:size(demos_in{n}.pos,2), demos_in{n}.pos, ...
                          linspace(1,size(demos_in{n}.pos,2),nbData));
        meanDemo = meanDemo + pos;
    end
    meanDemo = meanDemo / length(demos_in);

    amp          = endPos - startPos;
    rmseValues   = zeros(length(kRange), 1);
    maxErrValues = zeros(length(kRange), 1);
    allTrajs     = cell(length(kRange), 1);

    fprintf('K          RMSE      MaxError\n');

    for ki = 1:length(kRange)
        K = kRange(ki);

        try
            gmmData = dataTraining';
            for d = 1:numVarPos
                gmmData(:,d+1) = gmmData(:,d+1) / amp(d);
            end

            gmmodel = MixtureGaussians(gmmData, K);
            gmmodel = gmmodel.gmmFit(gmmData, 60, 1e-6, false);

            mask    = [1 zeros(1, numVarPos)];
            gmmodel = gmmodel.defineQueryDim(mask);
            gmmodel = gmmodel.gaussianMixtureRegression(xIn');

            fOut = gmmodel.regressedTraj(:,2:end)';
            fOut = fOut .* repmat(xIn,  numVarPos, 1) ...
                       .* repmat(amp,   1, nbData);

            xPos  = startPos;
            dx    = zeros(numVarPos, 1);
            rData = zeros(numVarPos, nbData);
            for t = 1:nbData
                ddx        = beta*(endPos-xPos) - alpha*dx + fOut(:,t);
                dx         = dx   + ddx*dt;
                xPos       = xPos + dx*dt;
                rData(:,t) = xPos;
            end

            err              = sqrt(sum((rData-meanDemo).^2, 1));
            rmseValues(ki)   = sqrt(mean(err.^2));
            maxErrValues(ki) = max(err);
            allTrajs{ki}     = rData;

            fprintf('%d          %.4f    %.4f\n', K, rmseValues(ki), maxErrValues(ki));

        catch
            fprintf('%d          FAILED\n', K);
            rmseValues(ki)   = NaN;
            maxErrValues(ki) = NaN;
            allTrajs{ki}     = [];
        end
    end

    [~, bestIdx] = min(rmseValues);
    fprintf('Best K = %d  RMSE = %.4f\n', kRange(bestIdx), rmseValues(bestIdx));

    % RMSE plot
    figure;
    plot(kRange, rmseValues, '-ob');
    xlabel('K (Number of Gaussians)'); ylabel('RMSE');
    title('GMM-GMR RMSE vs K');
    grid on;

    % Trajectory plot
    figure; hold on;
    for n = 1:length(demos_in)
        plot(demos_in{n}.pos(1,:), demos_in{n}.pos(2,:), '.', ...
             'Color', [.8 .8 .8], 'HandleVisibility', 'off');
    end
    plot(meanDemo(1,:), meanDemo(2,:), '--k', 'DisplayName', 'Mean Demo');
    colors = jet(length(kRange));
    colors = max(colors - 0.2, 0);
    for ki = 1:length(kRange)
        if ~isempty(allTrajs{ki})
            plot(allTrajs{ki}(1,:), allTrajs{ki}(2,:), ...
                 'Color', colors(ki,:), ...
                 'DisplayName', ['K=' num2str(kRange(ki)) ...
                                 ' RMSE=' num2str(rmseValues(ki),'%.2f')]);
        end
    end
    legend('Location', 'best');
    xlabel('X'); ylabel('Y');
    title('GMM-GMR Trajectories for Different K Values');
    axis equal; grid on;
end




function compareKvaluesWLR(dataTraining, demos_in, xIn, nbData, kRange, beta, alpha, dt, endPos, startPos, numVarPos, numDemos)

    % Compute mean demo as reference
    meanDemo = zeros(2, nbData);
    for n = 1:length(demos_in)
        pos      = spline(1:size(demos_in{n}.pos,2), demos_in{n}.pos, ...
                          linspace(1,size(demos_in{n}.pos,2),nbData));
        meanDemo = meanDemo + pos;
    end
    meanDemo = meanDemo / length(demos_in);

    amp      = endPos - startPos;
    x_full   = repmat(xIn, 1, numDemos);
    f_target = dataTraining(2:end, :);

    rmseValues   = zeros(length(kRange), 1);
    maxErrValues = zeros(length(kRange), 1);
    allTrajs     = cell(length(kRange), 1);

    fprintf('nStates    RMSE      MaxError\n');

    for ki = 1:length(kRange)
        K     = kRange(ki);
        mu    = linspace(xIn(1), xIn(end), K);
        sigma = (mu(1) - mu(2))^2;

        psi = exp(-0.5 * (x_full - mu').^2 / sigma);
        H   = psi ./ (sum(psi,1) + eps);

        aWeight = zeros(numVarPos, K);
        for i = 1:K
            W = diag(H(i,:));
            for d = 1:numVarPos
                bX          = (x_full * amp(d))';
                tY          = f_target(d,:)';
                aWeight(d,i) = (bX'*W*bX) \ (bX'*W*tY);
            end
        end

        psi_s = exp(-0.5 * (xIn - mu').^2 / sigma);
        H_s   = psi_s ./ (sum(psi_s,1) + eps);
        fOut  = aWeight * H_s ...
             .* repmat(xIn, numVarPos, 1) ...
             .* repmat(amp, 1, nbData);

        xPos  = startPos;
        dx    = zeros(numVarPos, 1);
        rData = zeros(numVarPos, nbData);
        for t = 1:nbData
            ddx        = beta*(endPos-xPos) - alpha*dx + fOut(:,t);
            dx         = dx   + ddx*dt;
            xPos       = xPos + dx*dt;
            rData(:,t) = xPos;
        end

        err              = sqrt(sum((rData-meanDemo).^2, 1));
        rmseValues(ki)   = sqrt(mean(err.^2));
        maxErrValues(ki) = max(err);
        allTrajs{ki}     = rData;

        fprintf('%d          %.4f    %.4f\n', K, rmseValues(ki), maxErrValues(ki));
    end

    [~, bestIdx] = min(rmseValues);
    fprintf('Best K = %d  RMSE = %.4f\n', kRange(bestIdx), rmseValues(bestIdx));

    % RMSE plot
    figure;
    plot(kRange, rmseValues, '-ob');
    xlabel('nStates'); ylabel('RMSE');
    title('RMSE vs Number of RBF Bases');
    grid on;

    % Trajectory plot
    figure; hold on;
    for n = 1:length(demos_in)
        plot(demos_in{n}.pos(1,:), demos_in{n}.pos(2,:), '.', ...
             'Color', [.8 .8 .8], 'HandleVisibility', 'off');
    end
    plot(meanDemo(1,:), meanDemo(2,:), '--k', 'DisplayName', 'Mean Demo');
    colors = jet(length(kRange));
    for ki = 1:length(kRange)
        if ~isempty(allTrajs{ki})
            plot(allTrajs{ki}(1,:), allTrajs{ki}(2,:), ...
                 'Color', colors(ki,:), ...
                 'DisplayName', ['K=' num2str(kRange(ki)) ...
                                 ' RMSE=' num2str(rmseValues(ki),'%.2f')]);
        end
    end
    legend('Location', 'best');
    xlabel('X'); ylabel('Y');
    title('WLR Trajectories for Different nStates');
    axis equal; grid on;
end

function compareDemos(allDemos, nbData, beta, alpha, dt, numStates)
    demoRange = 1:length(allDemos);  

    % Compute mean demo as reference
    meanDemo = zeros(2, nbData);
    for n = 1:length(allDemos)
        pos      = spline(1:size(allDemos{n}.pos,2), allDemos{n}.pos, ...
                          linspace(1,size(allDemos{n}.pos,2),nbData));
        meanDemo = meanDemo + pos;
    end
    meanDemo = meanDemo / length(allDemos);

    rmseValues   = zeros(length(demoRange), 1);
    maxErrValues = zeros(length(demoRange), 1);
    allTrajs     = cell(length(demoRange), 1);

    fprintf('NumDemos   RMSE      MaxError\n');

    for ki = 1:length(demoRange)
        numDemos = demoRange(ki);
        demos_in = {allDemos{1:numDemos}};

        try
            dmp  = SkillGeneralisation(3, numStates, beta, alpha, dt);
            dmp  = dmp.canonicalSystemInitialisation(1.1, nbData);
            dmp  = dmp.trajectort2Forcing(demos_in);

            amp     = dmp.endPos - dmp.startPos;
            gmmData = dmp.dataTraining';
            for d = 1:dmp.numVarPos
                gmmData(:,d+1) = gmmData(:,d+1) / amp(d);
            end

            gmmodel = MixtureGaussians(gmmData, numStates);
            gmmodel = gmmodel.gmmFit(gmmData, 60, 1e-6, false);

            mask    = [1 zeros(1, dmp.numVarPos)];
            gmmodel = gmmodel.defineQueryDim(mask);
            gmmodel = gmmodel.gaussianMixtureRegression(dmp.xIn');

            fOut = gmmodel.regressedTraj(:,2:end)';
            fOut = fOut .* repmat(dmp.xIn, dmp.numVarPos, 1) ...
                       .* repmat(amp, 1, nbData);

            xPos  = dmp.startPos;
            dx    = zeros(dmp.numVarPos, 1);
            rData = zeros(dmp.numVarPos, nbData);
            for t = 1:nbData
                ddx        = beta*(dmp.endPos-xPos) - alpha*dx + fOut(:,t);
                dx         = dx   + ddx*dt;
                xPos       = xPos + dx*dt;
                rData(:,t) = xPos;
            end

            err              = sqrt(sum((rData-meanDemo).^2, 1));
            rmseValues(ki)   = sqrt(mean(err.^2));
            maxErrValues(ki) = max(err);
            allTrajs{ki}     = rData;

            fprintf('%d          %.4f    %.4f\n', numDemos, rmseValues(ki), maxErrValues(ki));

        catch
            fprintf('%d          FAILED\n', numDemos);
            rmseValues(ki)   = NaN;
            maxErrValues(ki) = NaN;
            allTrajs{ki}     = [];
        end
    end

    [~, bestIdx] = min(rmseValues);
    fprintf('Best = %d demos  RMSE = %.4f\n', demoRange(bestIdx), rmseValues(bestIdx));

    % RMSE plot
    figure;
    plot(demoRange, rmseValues, '-ob');
    xlabel('Number of Demonstrations'); ylabel('RMSE');
    title('GMM-GMR RMSE vs Number of Demonstrations');
    grid on;

    % Trajectory plot
    figure; hold on;
    for n = 1:length(allDemos)
        plot(allDemos{n}.pos(1,:), allDemos{n}.pos(2,:), '.', ...
             'Color', [.8 .8 .8], 'HandleVisibility', 'off');
    end
    plot(meanDemo(1,:), meanDemo(2,:), '--k', 'DisplayName', 'Mean Demo');
    colors = jet(length(demoRange));
    
    for ki = 1:length(demoRange)
        if ~isempty(allTrajs{ki})
            plot(allTrajs{ki}(1,:), allTrajs{ki}(2,:), ...
                 'Color', colors(ki,:), ...
                 'DisplayName', [num2str(demoRange(ki)) ' demos' ...
                                 ' RMSE=' num2str(rmseValues(ki),'%.2f')]);
        end
    end
    legend('Location', 'best');
    xlabel('X'); ylabel('Y');
    title('GMM-GMR Trajectories for Different Number of Demonstrations');
    axis equal; grid on;
end

function compareDemosWLR(allDemos, nbData, beta, alpha, dt, numStates)
    
    demoRange = 1:length(allDemos);

    % Compute mean demo as reference
    meanDemo = zeros(2, nbData);
    for n = 1:length(allDemos)
        pos      = spline(1:size(allDemos{n}.pos,2), allDemos{n}.pos, ...
                          linspace(1,size(allDemos{n}.pos,2),nbData));
        meanDemo = meanDemo + pos;
    end
    meanDemo = meanDemo / length(allDemos);

    rmseValues   = zeros(length(demoRange), 1);
    maxErrValues = zeros(length(demoRange), 1);
    allTrajs     = cell(length(demoRange), 1);

    fprintf('NumDemos   RMSE      MaxError\n');

    for ki = 1:length(demoRange)
        numDemos = demoRange(ki);
        demos_in = {allDemos{1:numDemos}};

        dmp  = SkillGeneralisation(3, numStates, beta, alpha, dt);
        dmp  = dmp.canonicalSystemInitialisation(1.1, nbData);
        dmp  = dmp.trajectort2Forcing(demos_in);

        x      = dmp.xIn;
        amp    = dmp.endPos - dmp.startPos;
        mu     = linspace(x(1), x(end), numStates);
        sigma  = (mu(1) - mu(2))^2;
        x_full = repmat(x, 1, numDemos);

        psi = exp(-0.5 * (x_full - mu').^2 / sigma);
        H   = psi ./ (sum(psi,1) + eps);

        f_target = dmp.dataTraining(2:end, :);
        aWeight  = zeros(dmp.numVarPos, numStates);
        for i = 1:numStates
            W = diag(H(i,:));
            for d = 1:dmp.numVarPos
                bX           = (x_full * amp(d))';
                tY           = f_target(d,:)';
                aWeight(d,i) = (bX'*W*bX) \ (bX'*W*tY);
            end
        end

        psi_s = exp(-0.5 * (x - mu').^2 / sigma);
        H_s   = psi_s ./ (sum(psi_s,1) + eps);
        fOut  = aWeight * H_s ...
             .* repmat(x,   dmp.numVarPos, 1) ...
             .* repmat(amp, 1, nbData);

        xPos  = dmp.startPos;
        dx    = zeros(dmp.numVarPos, 1);
        rData = zeros(dmp.numVarPos, nbData);
        for t = 1:nbData
            ddx        = beta*(dmp.endPos-xPos) - alpha*dx + fOut(:,t);
            dx         = dx   + ddx*dt;
            xPos       = xPos + dx*dt;
            rData(:,t) = xPos;
        end

        err              = sqrt(sum((rData-meanDemo).^2, 1));
        rmseValues(ki)   = sqrt(mean(err.^2));
        maxErrValues(ki) = max(err);
        allTrajs{ki}     = rData;

        fprintf('%d          %.4f    %.4f\n', numDemos, rmseValues(ki), maxErrValues(ki));
    end

    [~, bestIdx] = min(rmseValues);
    fprintf('Best = %d demos  RMSE = %.4f\n', demoRange(bestIdx), rmseValues(bestIdx));

    % RMSE plot
    figure;
    plot(demoRange, rmseValues, '-ob');
    xlabel('Number of Demonstrations'); ylabel('RMSE');
    title('RMSE vs Number of Demonstrations');
    grid on;

    % Trajectory plot
    figure; hold on;
    for n = 1:length(allDemos)
        plot(allDemos{n}.pos(1,:), allDemos{n}.pos(2,:), '.', ...
             'Color', [.8 .8 .8], 'HandleVisibility', 'off');
    end
    plot(meanDemo(1,:), meanDemo(2,:), '--k', 'DisplayName', 'Mean Demo');
    colors = jet(length(demoRange));
    for ki = 1:length(demoRange)
        if ~isempty(allTrajs{ki})
            plot(allTrajs{ki}(1,:), allTrajs{ki}(2,:), ...
                 'Color', colors(ki,:), ...
                 'DisplayName', [num2str(demoRange(ki)) ' demos' ...
                                 ' RMSE=' num2str(rmseValues(ki),'%.2f')]);
        end
    end
    legend('Location', 'best');
    xlabel('X'); ylabel('Y');
    title('WLR Trajectories for Different Number of Demonstrations');
    axis equal; grid on;
end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  


















    end
end
