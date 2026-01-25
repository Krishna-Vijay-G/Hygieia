'use client';

import { useEffect, useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { motion } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  User, 
  Mail, 
  Phone, 
  Calendar,
  Shield,
  ShieldOff,
  Activity,
  ArrowLeft,
  Heart,
  Ribbon,
  Droplets,
  Scan,
  UserCheck,
  UserX,
  Trash2
} from 'lucide-react';
import { useAuthStore } from '@/lib/store';
import { usersApi } from '@/lib/api';
import { formatDate, formatDateTime, getAnalysisTypeLabel, getRiskColor, getImageURL } from '@/lib/utils';
import Link from 'next/link';

const analysisIcons: Record<string, any> = {
  'heart-prediction': Heart,
  'diabetes-prediction': Droplets,
  'skin-diagnosis': Scan,
  'breast-prediction': Ribbon,
  'breast-diagnosis': Ribbon,
};

const analysisColors: Record<string, { bg: string; icon: string }> = {
  'heart-prediction': { bg: 'bg-red-100 dark:bg-red-900/30', icon: 'text-red-600 dark:text-red-400' },
  'diabetes-prediction': { bg: 'bg-orange-100 dark:bg-orange-900/30', icon: 'text-orange-600 dark:text-orange-400' },
  'skin-diagnosis': { bg: 'bg-teal-100 dark:bg-teal-900/30', icon: 'text-teal-600 dark:text-teal-400' },
  'breast-prediction': { bg: 'bg-pink-100 dark:bg-pink-900/30', icon: 'text-pink-600 dark:text-pink-400' },
  'breast-diagnosis': { bg: 'bg-fuchsia-100 dark:bg-fuchsia-900/30', icon: 'text-fuchsia-600 dark:text-fuchsia-400' },
};

export default function UserProfilePage() {
  const router = useRouter();
  const params = useParams();
  const userId = params.id as string;
  const queryClient = useQueryClient();
  const { user, isAuthenticated, isLoading: authLoading } = useAuthStore();
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  // Fetch user details
  const { data: userData, isLoading: userLoading } = useQuery({
    queryKey: ['admin-user', userId],
    queryFn: () => usersApi.getUser(userId),
    enabled: isAuthenticated && user?.is_admin && !!userId,
  });

  // Fetch user analyses
  const { data: analysesData, isLoading: analysesLoading } = useQuery({
    queryKey: ['admin-user-analyses', userId],
    queryFn: () => usersApi.getUserAnalyses(userId),
    enabled: isAuthenticated && user?.is_admin && !!userId,
  });

  // Admin action mutations
  const toggleAdminMutation = useMutation({
    mutationFn: () => usersApi.toggleAdmin(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-user', userId] });
      queryClient.invalidateQueries({ queryKey: ['admin-users'] });
    },
  });

  const toggleActiveMutation = useMutation({
    mutationFn: () => usersApi.toggleActive(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-user', userId] });
      queryClient.invalidateQueries({ queryKey: ['admin-users'] });
    },
  });

  const deleteUserMutation = useMutation({
    mutationFn: () => usersApi.deleteUser(userId),
    onSuccess: () => {
      router.push('/admin/users');
    },
  });

  if (authLoading || userLoading || !user?.is_admin) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500" />
      </div>
    );
  }

  const profileUser = userData?.user;

  if (!profileUser) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">
            User Not Found
          </h2>
          <Link href="/admin/users" className="btn btn-primary">
            Back to Users
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Back Button */}
        <Link
          href="/admin/users"
          className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-400 hover:text-primary-600 dark:hover:text-primary-400 mb-6"
        >
          <ArrowLeft className="w-5 h-5" />
          Back to Users
        </Link>

        {/* Profile Header */}
        <div className="relative mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card p-8"
          >
            <div className="flex flex-col md:flex-row items-start md:items-center gap-6">
              {/* Avatar */}
              {profileUser.avatar_url ? (
                <img
                  src={getImageURL(profileUser.avatar_url)}
                  alt={`${profileUser.first_name} ${profileUser.last_name}`}
                  className="w-24 h-24 rounded-2xl object-cover"
                />
              ) : (
                <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-primary-500 to-secondary-500 flex items-center justify-center">
                  <span className="text-4xl font-bold text-white">
                    {profileUser.first_name[0]}{profileUser.last_name[0]}
                  </span>
                </div>
              )}

              {/* Info */}
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <h1 className="text-3xl font-bold text-slate-900 dark:text-white">
                    {profileUser.first_name} {profileUser.last_name}
                  </h1>
                  {profileUser.is_owner ? (
                    <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400">
                      <Shield className="w-4 h-4" />
                      Owner
                    </span>
                  ) : profileUser.is_admin ? (
                    <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400">
                      <Shield className="w-4 h-4" />
                      Admin
                    </span>
                  ) : null}
                  {profileUser.is_active ? (
                    <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
                      Active
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400">
                      Inactive
                    </span>
                  )}
                  {/* Admin Actions - Only visible to owners */}
                  {user?.is_owner && (
                    <div className="flex items-center gap-2 ml-auto">
                      <button
                        onClick={() => toggleAdminMutation.mutate()}
                        disabled={toggleAdminMutation.isPending || profileUser.is_owner}
                        className={`flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-lg border transition-colors ${
                          profileUser.is_owner 
                            ? 'border-slate-200 dark:border-slate-700 text-slate-400 dark:text-slate-600 cursor-not-allowed opacity-50' 
                            : 'border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800'
                        }`}
                        title={profileUser.is_owner ? "Cannot modify owner permissions" : undefined}
                      >
                        {profileUser.is_admin ? (
                          <>
                            <ShieldOff className="w-4 h-4" />
                            Remove Admin
                          </>
                        ) : (
                          <>
                            <Shield className="w-4 h-4" />
                            Make Admin
                          </>
                        )}
                      </button>
                      <button
                        onClick={() => toggleActiveMutation.mutate()}
                        disabled={toggleActiveMutation.isPending || profileUser.is_owner}
                        className={`flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-lg border transition-colors ${
                          profileUser.is_owner 
                            ? 'border-slate-200 dark:border-slate-700 text-slate-400 dark:text-slate-600 cursor-not-allowed opacity-50' 
                            : 'border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800'
                        }`}
                        title={profileUser.is_owner ? "Cannot deactivate owner" : undefined}
                      >
                        {profileUser.is_active ? (
                          <>
                            <UserX className="w-4 h-4" />
                            Deactivate
                          </>
                        ) : (
                          <>
                            <UserCheck className="w-4 h-4" />
                            Activate
                          </>
                        )}
                      </button>
                      <button
                        onClick={() => setShowDeleteModal(true)}
                        disabled={profileUser.is_owner}
                        className={`flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-lg border transition-colors ${
                          profileUser.is_owner 
                            ? 'border-slate-200 dark:border-slate-700 text-slate-400 dark:text-slate-600 cursor-not-allowed opacity-50' 
                            : 'border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20'
                        }`}
                        title={profileUser.is_owner ? "Cannot delete owner" : undefined}
                      >
                        <Trash2 className="w-4 h-4" />
                        Delete User
                      </button>
                    </div>
                  )}
                </div>
              <p className="text-xl text-slate-500 dark:text-slate-400 mb-4">
                @{profileUser.username}
              </p>

              {/* Contact Info */}
              <div className="flex flex-wrap gap-6">
                <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400">
                  <Mail className="w-5 h-5 text-primary-500" />
                  <span>{profileUser.email}</span>
                </div>
                {profileUser.phone && (
                  <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400">
                    <Phone className="w-5 h-5 text-primary-500" />
                    <span>{profileUser.phone}</span>
                  </div>
                )}
                <div className="flex items-center gap-2 text-slate-600 dark:text-slate-400">
                  <Calendar className="w-5 h-5 text-primary-500" />
                  <span>Joined {formatDate(profileUser.created_at)}</span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
        </div>

        {/* Stats Cards: Dashboard-style layout */}
        {(() => {
          const counts: Record<string, number> = {
            'heart-prediction': analysesData?.counts?.['heart-prediction'] || 0,
            'diabetes-prediction': analysesData?.counts?.['diabetes-prediction'] || 0,
            'skin-diagnosis': analysesData?.counts?.['skin-diagnosis'] || 0,
            'breast-prediction': analysesData?.counts?.['breast-prediction'] || 0,
            'breast-diagnosis': analysesData?.counts?.['breast-diagnosis'] || 0,
          };

          return (
            <div className="mb-8">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="card p-6 mb-4"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-slate-500 dark:text-slate-400">Total Analyses</p>
                    <p className="text-2xl font-bold text-slate-900 dark:text-white">
                      {analysesData?.analyses?.length || 0}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-primary-100 dark:bg-primary-900/30">
                    <Activity className="w-6 h-6 text-primary-600 dark:text-primary-400" />
                  </div>
                </div>
              </motion.div>

              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-5 gap-4">
                {/* Heart */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.12 }} className="card p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-500 dark:text-slate-400">Heart</p>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">{counts['heart-prediction']}</p>
                    </div>
                    <div className="p-3 rounded-xl bg-red-100 dark:bg-red-900/30">
                      <Heart className="w-6 h-6 text-red-600 dark:text-red-400" />
                    </div>
                  </div>
                </motion.div>

                {/* Diabetes */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.14 }} className="card p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-500 dark:text-slate-400">Diabetes</p>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">{counts['diabetes-prediction']}</p>
                    </div>
                    <div className="p-3 rounded-xl bg-orange-100 dark:bg-orange-900/30">
                      <Droplets className="w-6 h-6 text-orange-600 dark:text-orange-400" />
                    </div>
                  </div>
                </motion.div>

                {/* Skin */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.16 }} className="card p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-500 dark:text-slate-400">Skin</p>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">{counts['skin-diagnosis']}</p>
                    </div>
                    <div className="p-3 rounded-xl bg-teal-100 dark:bg-teal-900/30">
                      <Scan className="w-6 h-6 text-teal-600 dark:text-teal-400" />
                    </div>
                  </div>
                </motion.div>

                {/* Breast Risk */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.18 }} className="card p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-500 dark:text-slate-400">Breast Risk</p>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">{counts['breast-prediction']}</p>
                    </div>
                    <div className="p-3 rounded-xl bg-pink-100 dark:bg-pink-900/30">
                      <Ribbon className="w-6 h-6 text-pink-600 dark:text-pink-400" />
                    </div>
                  </div>
                </motion.div>

                {/* Tissue Analysis */}
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="card p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-500 dark:text-slate-400">Breast Tissue</p>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white">{counts['breast-diagnosis']}</p>
                    </div>
                    <div className="p-3 rounded-xl bg-fuchsia-100 dark:bg-fuchsia-900/30">
                      <Ribbon className="w-6 h-6 text-fuchsia-600 dark:text-fuchsia-400" />
                    </div>
                  </div>
                </motion.div>
              </div>
            </div>
          );
        })()}

        {/* Analysis History */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card overflow-hidden"
        >
          <div className="p-6 border-b border-slate-100 dark:border-slate-700">
            <h2 className="text-xl font-bold text-slate-900 dark:text-white">
              Analysis History
            </h2>
          </div>

          {analysesLoading ? (
            <div className="p-12 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500 mx-auto" />
            </div>
          ) : analysesData?.analyses?.length === 0 ? (
            <div className="p-12 text-center text-slate-500">
              No analyses found for this user
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-slate-50 dark:bg-slate-800">
                  <tr>
                    <th className="px-6 py-4 text-left text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                      Result
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                      Confidence
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                      Date
                    </th>
                    <th className="px-6 py-4 text-right text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                      Action
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
                  {analysesData?.analyses?.map((analysis: any) => {
                    const Icon = analysisIcons[analysis.analysis_type] || Activity;
                    const colors = analysisColors[analysis.analysis_type] || { bg: 'bg-slate-100 dark:bg-slate-700', icon: 'text-slate-600 dark:text-slate-400' };
                    return (
                      <tr
                        key={analysis.id}
                        className="hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
                      >
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <div className={`p-2 rounded-lg ${colors.bg}`}>
                              <Icon className={`w-5 h-5 ${colors.icon}`} />
                            </div>
                            <span className="font-medium text-slate-900 dark:text-white">
                              {getAnalysisTypeLabel(analysis.analysis_type)}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <span className={`font-medium ${getRiskColor(analysis.risk_level || '')}`}>
                            {analysis.risk_level || 'N/A'}
                          </span>
                        </td>
                        <td className="px-6 py-4">
                          {analysis.confidence ? (
                            <div className="flex items-center gap-2">
                              <div className="w-24 h-2 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-primary-500 rounded-full"
                                  style={{ width: `${analysis.confidence * 100}%` }}
                                />
                              </div>
                              <span className="text-sm text-slate-600 dark:text-slate-400">
                                {(analysis.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          ) : (
                            <span className="text-slate-400">N/A</span>
                          )}
                        </td>
                        <td className="px-6 py-4 text-sm text-slate-600 dark:text-slate-400">
                          {formatDateTime(analysis.created_at)}
                        </td>
                        <td className="px-6 py-4 text-right">
                          <Link
                            href={`/analysis/result/${analysis.id}`}
                            className="text-primary-600 hover:text-primary-700 dark:text-primary-400 font-medium text-sm"
                          >
                            View
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </motion.div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white dark:bg-slate-800 rounded-2xl p-6 max-w-md w-full"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className={`p-2 rounded-lg ${
                profileUser.is_owner 
                  ? 'bg-amber-100 dark:bg-amber-900/30' 
                  : 'bg-red-100 dark:bg-red-900/30'
              }`}>
                <Trash2 className={`w-5 h-5 ${
                  profileUser.is_owner 
                    ? 'text-amber-600 dark:text-amber-400' 
                    : 'text-red-600 dark:text-red-400'
                }`} />
              </div>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                {profileUser.is_owner ? 'Cannot Delete Owner' : 'Delete User'}
              </h3>
            </div>
            <p className="text-slate-600 dark:text-slate-400 mb-6">
              {profileUser.is_owner ? (
                <>
                  <strong>{profileUser.first_name} {profileUser.last_name}</strong> (@{profileUser.username}) is a system owner and cannot be deleted. 
                  Owner accounts are protected to maintain system integrity.
                </>
              ) : (
                <>
                  Are you sure you want to delete <strong>{profileUser.first_name} {profileUser.last_name}</strong> (@{profileUser.username})? 
                  This action cannot be undone and will permanently remove all their data.
                </>
              )}
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setShowDeleteModal(false)}
                className="flex-1 btn-secondary"
              >
                {profileUser.is_owner ? 'Close' : 'Cancel'}
              </button>
              {!profileUser.is_owner && (
                <button
                  onClick={() => {
                    deleteUserMutation.mutate();
                    setShowDeleteModal(false);
                  }}
                  disabled={deleteUserMutation.isPending}
                  className="flex-1 btn btn-primary px-6 bg-red-600 hover:bg-red-700 dark:bg-red-600 dark:hover:bg-red-700"
                >
                  {deleteUserMutation.isPending ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                      Deleting...
                    </>
                  ) : (
                    <>
                      <Trash2 className="w-5 h-5 mr-2" />
                      Delete User
                    </>
                  )}
                </button>
              )}
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
