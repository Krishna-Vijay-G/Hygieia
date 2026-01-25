'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import toast from 'react-hot-toast';
import { Eye, EyeOff, Mail, Lock, User, Phone, Loader2, CheckCircle } from 'lucide-react';
import { useAuthStore } from '@/lib/store';

const registerSchema = z.object({
  first_name: z.string().min(2, 'First name must be at least 2 characters'),
  last_name: z.string().min(1, 'Last name must be at least 1 characters'),
  username: z.string()
    .min(3, 'Username must be at least 3 characters')
    .regex(/^[a-zA-Z0-9_]+$/, 'Username can only contain letters, numbers, and underscores')
    .transform((val) => val.toLowerCase()),
  email: z.string().email('Invalid email address'),
  phone: z.string().optional(),
  password: z.string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
    .regex(/[a-z]/, 'Password must contain at least one lowercase letter')
    .regex(/[0-9]/, 'Password must contain at least one number'),
  confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ['confirmPassword'],
});

type RegisterForm = z.infer<typeof registerSchema>;

export default function RegisterPage() {
  const router = useRouter();
  const { register: registerUser } = useAuthStore();
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm<RegisterForm>({
    resolver: zodResolver(registerSchema),
  });

  const password = watch('password', '');
  
  const passwordRequirements = [
    { text: 'At least 8 characters', met: password.length >= 8 },
    { text: 'One uppercase letter', met: /[A-Z]/.test(password) },
    { text: 'One lowercase letter', met: /[a-z]/.test(password) },
    { text: 'One number', met: /[0-9]/.test(password) },
  ];

  const onSubmit = async (data: RegisterForm) => {
    setIsLoading(true);
    try {
      const { confirmPassword, ...registerData } = data;
      await registerUser(registerData);
      toast.success('Account created successfully!');
      router.push('/dashboard');
    } catch (error: any) {
      toast.error(error.response?.data?.error || 'Registration failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center py-12 px-4 bg-gradient-to-br from-slate-50 to-primary-50 dark:from-slate-900 dark:to-slate-800">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-lg"
      >
        <div className="glass-card p-8">
          {/* Header */}
          <div className="text-center mb-8">
            <Link href="/" className="inline-flex items-center gap-2 mb-6 justify-center">
              <img 
                src="/hygieia_color.png" 
                alt="Hygieia Logo" 
                className="h-16 w-auto"
              />
            </Link>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
              Create your account
            </h1>
            <p className="text-slate-600 dark:text-slate-400 mt-2">
              Join Hygieia for AI-powered health insights
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  First Name
                </label>
                <input
                  type="text"
                  {...register('first_name')}
                  className={`input ${errors.first_name ? 'input-error' : ''}`}
                  placeholder="Jackie"
                />
                {errors.first_name && (
                  <p className="mt-1 text-sm text-red-500">{errors.first_name.message}</p>
                )}
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                  Last Name
                </label>
                <input
                  type="text"
                  {...register('last_name')}
                  className={`input ${errors.last_name ? 'input-error' : ''}`}
                  placeholder="Chan"
                />
                {errors.last_name && (
                  <p className="mt-1 text-sm text-red-500">{errors.last_name.message}</p>
                )}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Username
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type="text"
                  {...register('username')}
                  className={`input pl-10 ${errors.username ? 'input-error' : ''}`}
                  placeholder="jackie"
                />
              </div>
              <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                Username will be stored in lowercase
              </p>
              {errors.username && (
                <p className="mt-1 text-sm text-red-500">{errors.username.message}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Email
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type="email"
                  {...register('email')}
                  className={`input pl-10 ${errors.email ? 'input-error' : ''}`}
                  placeholder="jackie@email.com"
                />
              </div>
              {errors.email && (
                <p className="mt-1 text-sm text-red-500">{errors.email.message}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Phone (optional)
              </label>
              <div className="relative">
                <Phone className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type="tel"
                  {...register('phone')}
                  className="input pl-10"
                  placeholder="+91 9876543210"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  {...register('password')}
                  className={`input pl-10 pr-10 ${errors.password ? 'input-error' : ''}`}
                  placeholder="Create a strong password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
              {/* Password requirements */}
              <div className="mt-2 grid grid-cols-2 gap-2">
                {passwordRequirements.map((req) => (
                  <div
                    key={req.text}
                    className={`flex items-center gap-1.5 text-xs ${
                      req.met ? 'text-green-600' : 'text-slate-400'
                    }`}
                  >
                    <CheckCircle className={`w-3.5 h-3.5 ${req.met ? 'text-green-500' : ''}`} />
                    {req.text}
                  </div>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Confirm Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  {...register('confirmPassword')}
                  className={`input pl-10 ${errors.confirmPassword ? 'input-error' : ''}`}
                  placeholder="Confirm your password"
                />
              </div>
              {errors.confirmPassword && (
                <p className="mt-1 text-sm text-red-500">{errors.confirmPassword.message}</p>
              )}
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full btn-primary btn-lg"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin mr-2" />
                  Creating account...
                </>
              ) : (
                'Create account'
              )}
            </button>
          </form>

          {/* Footer */}
          <p className="mt-6 text-center text-sm text-slate-600 dark:text-slate-400">
            Already have an account?{' '}
            <Link
              href="/login"
              className="text-primary-600 hover:text-primary-700 dark:text-primary-400 font-medium"
            >
              Sign in
            </Link>
          </p>
        </div>
      </motion.div>
    </div>
  );
}
