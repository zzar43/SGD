function compute_gb(u, v, model_coef)
    gb = zeros(model_coef.Nx, model_coef.Ny, model_coef.Nt)
    part1 = (v[3:end,2:end-1,:]-v[2:end-1,2:end-1,:]) .* (u[3:end,2:end-1,:]-u[2:end-1,2:end-1,:]) ./ (model_coef.dx^2)
    part2 = (v[2:end-1,3:end,:]-v[2:end-1,2:end-1,:]) .* (u[2:end-1,3:end,:]-u[2:end-1,2:end-1,:]) ./ (model_coef.dy^2)
    gb[2:end-1,2:end-1,:] = part1 + part2
    
    gb = sum(gb, dims=3);
    gb = gb[:,:,1]
    return gb
end

function compute_ga(u, v, model_coef)
    uu = similar(u);
    uu[:,:,2:end-1] = (u[:,:,1:end-2] - 2 .* u[:,:,2:end-1] + u[:,:,3:end]) ./ (model_coef.dt^2)
    uu = uu[:,:,end:-1:1]
    
    ga = v .* uu
    ga = sum(ga, dims=3)
    ga = ga[:,:,1]
    return ga
end

function compute_gradient(a0, b0, received_data, model_coef, receiver_position, source_position, source_func)
    
    d0, u0 = acoustic_solver_all_sources(a0, b0, model_coef, source_position, source_func, receiver_position);
    println("Reference model computed.")
    
#     Initialize
    Ns = size(source_position,1)
    ga = zeros(model_coef.Nx, model_coef.Ny)
    gb = zeros(model_coef.Nx, model_coef.Ny)
    
    print("Computing source: ")
    for iter = 1:Ns
        adjoint_source = d0[:,:,iter] - received_data[:,:,iter]
        adjoint_source = adjoint_source[end:-1:1,:];
        d1, v = acoustic_solver(a0, b0, model_coef, receiver_position, adjoint_source, receiver_position; source_index=0)

        gb0 = compute_gb(u0[:,:,:,iter], v, model_coef)
        ga0 = compute_ga(u0[:,:,:,iter], v, model_coef)

        ga = ga + ga0
        gb = gb + gb0

        print(iter, ", ")
    end
    println("Done.")
    
    ga = -1 * ga
    gb = -1 * gb
    
    return ga, gb
end

# output a row vector
function compute_gradient_lbfgsb(a0, b0, received_data, model_coef, source_position, source_func, receiver_position)
    
    d0, u0 = acoustic_solver_all_sources(a0, b0, model_coef, source_position, source_func, receiver_position);
    println("Reference model computed.")
    
#     Initialize
    Ns = size(source_position,1)
    ga = zeros(model_coef.Nx, model_coef.Ny)
    gb = zeros(model_coef.Nx, model_coef.Ny)
    f = 0;
    
    print("Computing source: ")
    for iter = 1:Ns
        adjoint_source = d0[:,:,iter] - received_data[:,:,iter]
        adjoint_source = adjoint_source[end:-1:1,:];
        d1, v = acoustic_solver(a0, b0, model_coef, receiver_position, adjoint_source, receiver_position; source_index=0)

        gb0 = compute_gb(u0[:,:,:,iter], v, model_coef)
        ga0 = compute_ga(u0[:,:,:,iter], v, model_coef)

        ga = ga + ga0
        gb = gb + gb0
        f = f + norm(adjoint_source).^2;

        print(iter, ", ")
    end
    println("Done.")
    
    g = [ga[:]; gb[:]]
#     g = g ./ norm(g);
    g = -1 * g;
    return f, g
end

using LinearAlgebra
function lbfgsb_func(x, received_data, model_coef, source_position, source_func, receiver_position)
    
#     x should be a row vector with size 2 * Nx * Ny
    if size(x,1) != 2 * model_coef.Nx * model_coef.Ny
        error("Please check the size of x.")
    else
        a0 = x[1:model_coef.Nx*model_coef.Ny]
        a0 = reshape(a0, model_coef.Nx, model_coef.Ny)
        b0 = x[model_coef.Nx*model_coef.Ny+1:end]
        b0 = reshape(b0, model_coef.Nx, model_coef.Ny)
    end
    
    f, g = compute_gradient_lbfgsb(a0, b0, received_data, model_coef, source_position, source_func, receiver_position)
    g = g ./ maximum(abs.(g))
    
    return f, g
end

using LBFGSB
function lbfgs_fwi(a_true, b_true, a0, b0, model_coef, source_position, source_func, receiver_position; iterTime=10)
#     setup
    n = model_coef.Nx*model_coef.Ny * 2;
    optimizer = L_BFGS_B(n, 5);
    bounds = zeros(3, n)
    low_a = minimum(a_true);
    low_b = minimum(b_true)
    up_a = maximum(a_true)
    up_b = maximum(b_true)
    for i = 1:model_coef.Nx*model_coef.Ny
        bounds[1,i] = 2
        bounds[2,i] = low_a
        bounds[3,i] = up_a
    end
    for i = model_coef.Nx*model_coef.Ny+1 : n
        bounds[1,i] = 2
        bounds[2,i] = low_b
        bounds[3,i] = up_b
    end
    
    received_data, u = acoustic_solver_all_sources(a_true, b_true, model_coef, source_position, source_func, receiver_position);
    u = [];
    println("Received data computed.")
    
#     setup function
    func(x) = lbfgsb_func(x, received_data, model_coef, source_position, source_func, receiver_position)
    
    x0 = [a0[:]; b0[:]];
    
    fout, xout = optimizer(func, x0, bounds, m=5, factr=1e12, pgtol=1e-10, iprint=1, maxfun=10000, maxiter=iterTime)
    
    a1 = xout[1:model_coef.Nx*model_coef.Ny];
    a1 = reshape(a1, model_coef.Nx, model_coef.Ny);
    b1 = xout[model_coef.Nx*model_coef.Ny+1:end];
    b1 = reshape(b1, model_coef.Nx, model_coef.Ny);
    
    return a1, b1
end



# ======================================================
using Distributed
using SharedArrays

function acoustic_solver_all_sources_parallel(a, b, model_coef, source_position, source_func, receiver_position)
    Ns = size(source_position,1)
#     received_data = zeros(model_coef.Nt, size(receiver_position,1), Ns);
#     wavefield = zeros(model_coef.Nx, model_coef.Ny, model_coef.Nt, Ns)
    received_data = SharedArray{Float64}(model_coef.Nt, size(receiver_position,1), Ns);
    wavefield = SharedArray{Float64}(model_coef.Nx, model_coef.Ny, model_coef.Nt, Ns);
    
    @sync @distributed for iter = 1:Ns
        data, u = acoustic_solver(a, b, model_coef, source_position[iter,:], source_func, receiver_position);
        received_data[:,:,iter] = data;
        wavefield[:,:,:,iter] = u;
        println(iter)
    end
    
    return received_data, wavefield
end

function compute_gradient_parallel(a0, b0, received_data, model_coef, receiver_position, source_position, source_func)
    
    d0, u0 = acoustic_solver_all_sources_parallel(a0, b0, model_coef, source_position, source_func, receiver_position);
    println("Reference model computed.")
    
    # Initialize
    Ns = size(source_position,1)
    ga = SharedArray{Float64}(model_coef.Nx, model_coef.Ny, Ns)
    gb = SharedArray{Float64}(model_coef.Nx, model_coef.Ny, Ns)
    
    print("Computing source: ")
    @sync @distributed for iter = 1:Ns
        adjoint_source = received_data[:,:,iter] - d0[:,:,iter]
        adjoint_source = adjoint_source[end:-1:1,:];
        d1, v = acoustic_solver(a0, b0, model_coef, receiver_position, adjoint_source, receiver_position; source_index=0)

        gb0 = compute_gb(u0[:,:,:,iter], v, model_coef)
        ga0 = compute_ga(u0[:,:,:,iter], v, model_coef)
        ga[:,:,iter] = ga0;
        gb[:,:,iter] = gb0;

        print(iter, ", ")
    end
    println("Done.")
    ga = sum(ga, dims=3);
    gb = sum(gb, dims=3);
    ga = ga[:,:,1];
    gb = gb[:,:,1];

    return ga, gb
end

using LinearAlgebra
function lbfgsb_func_parallel(x, received_data, model_coef, source_position, source_func, receiver_position)
    
#     x should be a row vector with size 2 * Nx * Ny
    if size(x,1) != 2 * model_coef.Nx * model_coef.Ny
        error("Please check the size of x.")
    else
        a0 = x[1:model_coef.Nx*model_coef.Ny]
        a0 = reshape(a0, model_coef.Nx, model_coef.Ny)
        b0 = x[model_coef.Nx*model_coef.Ny+1:end]
        b0 = reshape(b0, model_coef.Nx, model_coef.Ny)
    end


    
    f, g = compute_gradient_lbfgsb(a0, b0, received_data, model_coef, source_position, source_func, receiver_position)
    g = g ./ maximum(abs.(g))
    
    return f, g
end


function lbfgs_fwi_parallel(a_true, b_true, a0, b0, model_coef, source_position, source_func, receiver_position; iterTime=10)
    #     setup
        n = model_coef.Nx*model_coef.Ny * 2;
        optimizer = L_BFGS_B(n, 5);
        bounds = zeros(3, n)
        low_a = minimum(a_true);
        low_b = minimum(b_true)
        up_a = maximum(a_true)
        up_b = maximum(b_true)
        for i = 1:model_coef.Nx*model_coef.Ny
            bounds[1,i] = 2
            bounds[2,i] = low_a
            bounds[3,i] = up_a
        end
        for i = model_coef.Nx*model_coef.Ny+1 : n
            bounds[1,i] = 2
            bounds[2,i] = low_b
            bounds[3,i] = up_b
        end
        
        received_data, u = acoustic_solver_all_sources(a_true, b_true, model_coef, source_position, source_func, receiver_position);
        u = [];
        println("Received data computed.")
        
    #     setup function
        func(x) = lbfgsb_func(x, received_data, model_coef, source_position, source_func, receiver_position)
        
        x0 = [a0[:]; b0[:]];
        
        fout, xout = optimizer(func, x0, bounds, m=5, factr=1e12, pgtol=1e-10, iprint=1, maxfun=10000, maxiter=iterTime)
        
        a1 = xout[1:model_coef.Nx*model_coef.Ny];
        a1 = reshape(a1, model_coef.Nx, model_coef.Ny);
        b1 = xout[model_coef.Nx*model_coef.Ny+1:end];
        b1 = reshape(b1, model_coef.Nx, model_coef.Ny);
        
        return a1, b1
    end