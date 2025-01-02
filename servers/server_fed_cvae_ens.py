import copy

from servers.server_fed_cvae_kd import ServerFedCVAEKD
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
import numpy as np
import torch
from utils import (WrapperDataset, WrapperDatasetG, average_weights, one_hot_encode,
                   reconstruction_loss)
from skimage.metrics import structural_similarity as ssim
import skfuzzy as fuzz
from joblib import Parallel, delayed
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import kl_div
from torch.utils.data import DataLoader


class ServerFedCVAEEns(ServerFedCVAEKD):
    def __init__(
        self,
        base_params,
        z_dim,
        image_size,
        beta,
        classifier_num_train_samples,
        classifier_epochs,
        uniform_range,
        should_weight,
        should_initialize_same,
        heterogeneous_models,
        trial,
        fuzzy_v,
        meta_beta,
        inner_loop,
        log,
    ):
        super().__init__(
            base_params,
            z_dim,
            image_size,
            beta,
            classifier_num_train_samples,
            classifier_epochs,
            None,
            None,
            0.01,
            uniform_range,
            should_weight,
            should_initialize_same,
            0,
            0,
            heterogeneous_models,
            0,
            trial
        )
        self.base_params = base_params
        self.fuzzy_v = fuzzy_v
        self.meta_beta = meta_beta
        self.inner_loop = inner_loop

    # 获取decoder的参数并展平成向量
    def get_flat_params(self, model):
        return torch.cat([p.flatten() for p in model.parameters()])

    # 计算Wasserstein距离
    def calculate_wasserstein_scores(self, selected_users):
        decoders = [user.model.decoder for user in selected_users]
        num = len(selected_users)
        scores = []
        # 计算Wasserstein距离矩阵
        wasserstein_matrix = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                if i != j:
                    params1 = self.get_flat_params(decoders[i]).detach().cpu().numpy()
                    params2 = self.get_flat_params(decoders[j]).detach().cpu().numpy()
                    wasserstein_matrix[i][j] = wasserstein_distance(params1, params2)
        # 对每个decoder计算Wasserstein距离得分
        for i in range(num):
            distances = wasserstein_matrix[i][wasserstein_matrix[i] != 0]

            # 进行2-median聚类
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(distances.reshape(-1, 1))

            # 找到majority group
            labels = kmeans.labels_
            majority_group = distances[labels == np.argmax(np.bincount(labels))]

            # 计算均值
            score = np.mean(majority_group)
            scores.append(score)
        # 返回每个decoder的wasserstein得分
        return scores

    # 计算余弦距离
    def calculate_cosine_scores(self, selected_users):
        decoders = [user.model.decoder for user in selected_users]
        num = len(selected_users)
        scores = []
        # 计算cosine距离矩阵
        cosine_matrix = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                if i != j:
                    params1 = self.get_flat_params(decoders[i]).detach().cpu().numpy()
                    params2 = self.get_flat_params(decoders[j]).detach().cpu().numpy()
                    cosine_matrix[i][j] = cosine(params1, params2)
        # 对每个decoder计算cosine距离得分
        for i in range(num):
            distances = cosine_matrix[i][cosine_matrix[i] != 0]

            # 进行2-median聚类
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(distances.reshape(-1, 1))

            # 找到majority group
            labels = kmeans.labels_
            majority_group = distances[labels == np.argmax(np.bincount(labels))]

            # 计算均值
            score = np.mean(majority_group)
            scores.append(score)
        # 返回每个decoder的cosine得分
        return scores

    # 每个decoder生成数据
    def generate_data_from_decoders(self, users, num_train_samples):
        X_vals = torch.Tensor()
        y_vals = torch.Tensor()
        # z_vals = torch.Tensor()

        total_train_samples = 0
        count_user = 0
        for u in users:
            u.model.eval()

            if self.should_weight:
                user_num_train_samples = int(u.data_amt * num_train_samples)
            else:
                user_num_train_samples = int(num_train_samples / len(users))

            if count_user == len(users) - 1:
                user_num_train_samples = num_train_samples - total_train_samples
            else:
                total_train_samples += user_num_train_samples
                count_user += 1
            z = u.model.sample_z(
                int(user_num_train_samples), "truncnorm", width=self.uniform_range
            ).to(self.device)

            classes = np.arange(u.num_classes)
            y = torch.from_numpy(
                np.random.choice(classes, size=user_num_train_samples, p=u.pmf)
            )

            y_hot = one_hot_encode(y, u.num_classes).to(self.device)

            _, generated_image = u.model.decoder(z, y_hot)
            # X = generated_image.detach()
            X = generated_image
            X, y_hot = X.cpu(), y_hot.cpu()
            X_vals = torch.cat((X_vals, X), 0)
            y_vals = torch.cat((y_vals, y_hot), 0)
            # z_vals = torch.cat((z_vals, z), 0)

        X_vals = torch.sigmoid(X_vals)

        return X_vals, y_vals  # 返回生成的数据和标签

    # 计算 SSIM损失 加速版本；用于三维特征向量
    def calculate_ssim_scores(self, users, classifier_num_train_samples, dataset):
        all_generated_data = {}  # 存储所有生成数据的字典，键为类别

        # 存储所有用户的生成数据
        for u in users:
            generated_data, labels = self.generate_data_from_decoders([u], classifier_num_train_samples)
            for i in range(generated_data.size(0)):
                category = labels[i].argmax().item()  # 获取类别
                if category not in all_generated_data:
                    all_generated_data[category] = []
                all_generated_data[category].append(generated_data[i].detach().cpu().numpy())
        scores = []

        # 定义计算单个样本SSIM的函数
        def compute_ssim_for_sample(generated_sample, all_data_in_category):
            category_losses = []
            for other_data in all_data_in_category:
                if dataset in ['mnist', 'fmnist']:
                    ssim_value = ssim(generated_sample.squeeze(),
                                      other_data.squeeze(),
                                      data_range=1, multichannel=False)
                elif dataset == 'svhn':
                    ssim_value = ssim(generated_sample.transpose(1, 2, 0),
                                      other_data.transpose(1, 2, 0),
                                      data_range=1, multichannel=True)
                else:
                    raise ValueError("dataset not in [mnist, fmnist, svhn]")
                category_losses.append(1 - ssim_value)
            return np.mean(category_losses) if category_losses else 0

        # 对每个decoder计算SSIM得分
        for u in users:
            generated_data, labels = self.generate_data_from_decoders([u], classifier_num_train_samples)
            ssim_losses = Parallel(n_jobs=-1)(delayed(compute_ssim_for_sample)(
                generated_data[i].detach().cpu().numpy(),
                all_generated_data[labels[i].argmax().item()]
            ) for i in range(generated_data.size(0)))

            # 计算所有类别的平均SSIM损失
            final_score = np.mean(ssim_losses) if ssim_losses else 0
            scores.append(final_score)

        return scores
    # #计算 SSIM损失  原始版本
    # def calculate_ssim_scores(self, users, classifier_num_train_samples, dataset):
    #     all_generated_data = {}  # 存储所有生成数据的字典，键为类别
    #
    #     # 存储所有用户的生成数据
    #     for u in users:
    #         generated_data, labels = self.generate_data_from_decoders([u], classifier_num_train_samples)
    #         for i in range(generated_data.size(0)):
    #             category = labels[i].argmax().item()  # 获取类别
    #             if category not in all_generated_data:
    #                 all_generated_data[category] = []
    #             all_generated_data[category].append(generated_data[i].cpu().numpy())
    #     scores = []
    #
    #     # 对每个decoder计算SSIM得分
    #     for u in users:
    #         generated_data, labels = self.generate_data_from_decoders([u], classifier_num_train_samples)
    #         ssim_losses = []
    #
    #         for i in range(generated_data.size(0)):
    #             category = labels[i].argmax().item()
    #             category_losses = []
    #
    #             # 计算当前decoder与其他decoder同类别生成数据之间的SSIM损失
    #             for other_category in all_generated_data:
    #                 if other_category == category:
    #                     for other_data in all_generated_data[other_category]:
    #                         # 计算SSIM损失,根据数据通道数类型
    #                         if dataset in ['mnist', 'fmnist']:
    #                             ssim_value = 1-ssim(generated_data[i].squeeze(),
    #                                               other_data.squeeze(),
    #                                               data_range=1, multichannel=False)
    #                         elif dataset == 'svhn':  # 三通道彩色图像
    #                             ssim_value = 1-ssim(generated_data[i].transpose(1, 2, 0),
    #                                               other_data.transpose(1, 2, 0),
    #                                               data_range=1, multichannel=True)
    #                         else:
    #                             raise ValueError("dataset not in [mnist,fmnist,svhn]")
    #                         # ssim_loss = 1 - ssim_value  # 使用1 - SSIM
    #                         category_losses.append(ssim_value) #记录与当前样本相同类别的所有loss
    #
    #             # 计算该样本的平均损失
    #             if category_losses:
    #                 average_loss = np.mean(category_losses) #当前样本的所有loss平均值
    #                 ssim_losses.append(average_loss)#记录所有样本的loss，每个样本对应一个loss
    #
    #         # 计算所有类别的平均SSIM损失
    #         if ssim_losses:
    #             final_score = np.mean(ssim_losses)#当前user的loss
    #         else:
    #             final_score = 0
    #
    #         scores.append(final_score)
    #
    #     return scores

    #度量值归一化，Min-Max 归一化函数
    def min_max_normalize(self, values):
        min_val = np.min(values)
        max_val = np.max(values)
        # 防止出现除零错误
        if max_val - min_val == 0:
            return np.zeros_like(values)
        return (values - min_val) / (max_val - min_val)

# 恶意簇和模糊簇的纠正
    def correct_malicious_decoders(self, users, benign_decoders, malicious_decoders, uncertain_decoders, meta_beta,
                                   maml_lr=0.01, outer_steps=1):
        for malicious_decoder_index in malicious_decoders + uncertain_decoders:

            # 初始化Adam优化器
            optimizer = optim.Adam(users[malicious_decoder_index].model.decoder.parameters(), lr=maml_lr)
            # 外循环
            for outer_step in range(outer_steps):
                meta_loss = torch.zeros(1, requires_grad=True, device=self.device)
            # 内循环：针对恶意decoder的每个生成类别进行纠正
                for cls, proportion in enumerate(users[malicious_decoder_index].pmf):
                    if proportion > 0:
                        # 获取同类别的良性decoders的索引
                        benign_index_for_class = [decoder for decoder in benign_decoders if users[decoder].pmf[cls] > 0]
                        # 检查是否有良性解码器
                        if len(benign_index_for_class) == 0:
                            print(f"No benign decoders for class {cls}. Skipping this class.")
                            continue

                        malicious_loss = torch.zeros(1, requires_grad=True, device=self.device)
                        # 计算SSIM和KL损失
                        for benign_decoder in benign_index_for_class:
                            if self.should_weight:
                                user_num_train_samples = int(users[benign_decoder].data_amt * self.classifier_num_train_samples)
                            else:
                                user_num_train_samples = int(self.classifier_num_train_samples / len(users))
                            if int(user_num_train_samples * users[benign_decoder].pmf[cls]) == 0:
                                benign_index_for_class.remove(benign_decoder)
                                continue
                            task_loss = torch.zeros(1, requires_grad=True, device=self.device)
                            # 内循环优化：更新恶意decoder
                            for step in range(self.inner_loop):
                                optimizer.zero_grad()
                                # SSIM损失
                                ssim_loss = self.compute_ssim_loss(users[malicious_decoder_index], users[benign_decoder], cls)

                                # KL散度损失
                                # kl_loss = self.compute_kl_loss(users[malicious_decoder_index], users[benign_decoder], cls, len(users),
                                #                                self.classifier_num_train_samples)
                                # 组合损失
                                # task_loss = meta_beta * ssim_loss + (1-meta_beta) * kl_loss
                                task_loss = ssim_loss
                                # 内循环优化：更新恶意decoder
                                task_loss.backward(retain_graph=True)
                                optimizer.step()

                            malicious_loss = malicious_loss + task_loss.detach()
                        if benign_index_for_class:
                            malicious_loss = malicious_loss/len(benign_index_for_class)
                    # 更新元损失
                    meta_loss = meta_loss + malicious_loss

                # 外循环优化：基于元任务损失更新恶意decoders
                optimizer.zero_grad()
                meta_loss.backward(retain_graph=True)
                optimizer.step()
        # return malicious_decoders

# 计算KL损失用的，生成特征图
    def generate_feature_from_decoders(self, u, cls, user_num, num_train_samples):
        F_vals = torch.Tensor()
        # y_vals = torch.Tensor()
        # z_vals = torch.Tensor()

        total_train_samples = 0
        # count_user = 0
        # for u in users:
        u.model.eval()

        if self.should_weight:
            user_num_train_samples = int(u.data_amt * num_train_samples)
        else:
            user_num_train_samples = int(num_train_samples / user_num)
        #
        # if count_user == user_num - 1:
        #     user_num_train_samples = num_train_samples - total_train_samples
        # else:
        #     total_train_samples += user_num_train_samples
        #     count_user += 1

        print(f"PMF[{cls}] for user: {u.pmf[cls]}")  # 打印pmf值
        print(f"Generating {int(user_num_train_samples * u.pmf[cls])} samples for class {cls}")  # 打印生成样本数
        if int(user_num_train_samples * u.pmf[cls]) == 0:
            print(f"比例过小，取整后为0，该user没有样本生成{cls}类别特征")
            return torch.tensor([]).to(self.device)
        z = u.model.sample_z(
            int(user_num_train_samples * u.pmf[cls]), "truncnorm", width=self.uniform_range
        ).to(self.device)
        if z.size(0) == 0:
            print(f"No latent variables sampled for class {cls}")
            return torch.tensor([]).to(self.device)
        # classes = np.arange(u.num_classes)
        # y = torch.from_numpy(
        #     np.random.choice(classes, size=user_num_train_samples * u.pmf[cls], p=u.pmf)
        # )
        y = torch.full((int(user_num_train_samples * u.pmf[cls]),), cls, dtype=torch.long)
        y = y.cpu()
        y_hot = one_hot_encode(y, u.num_classes).to(self.device)

        feature_maps, _ = u.model.decoder(z, y_hot)
        F = feature_maps.detach()
        F_vals = torch.cat((F_vals, F.cpu()), 0)

        # y_vals = torch.cat((y_vals, y_hot), 0)
        # z_vals = torch.cat((z_vals, z), 0)

        # F_vals = torch.sigmoid(F_vals) 好像不需要归一化

        return F_vals  # 返回特征图

    # 计算KL散度损失 元学习中损失
    def compute_kl_loss(self, malicious_decoder, benign_decoder, cls, user_num, num_train_samples):
        # 获取恶意和良性decoder的最后一个反卷积层前的特征图
        benign_features = self.generate_feature_from_decoders(benign_decoder, cls, user_num, num_train_samples)
        malicious_features = self.generate_feature_from_decoders(malicious_decoder, cls, user_num, num_train_samples)
        benign_features.requires_grad_(True)
        malicious_features.requires_grad_(True)
        # 计算KL散度 (假设benign_features和malicious_features的形状匹配)
        if benign_features.size(0) == 0 or malicious_features.size(0) == 0:
            print(f"Skipping KL loss for class {cls}: empty feature maps")
            return torch.zeros(1, device=self.device, requires_grad=True)
        # 裁剪到最小大小
        min_size = min(benign_features.size(0), malicious_features.size(0))
        benign_features = benign_features[:min_size]
        malicious_features = malicious_features[:min_size]
        kl_loss = F.kl_div(
            F.log_softmax(benign_features, dim=-1),  # 计算良性解码器特征图的对数softmax
            F.softmax(malicious_features, dim=-1),  # 计算恶意解码器特征图的softmax
            reduction="batchmean"  # 默认按照batch取平均
        ).requires_grad_()
        print(f"KL Loss for class {cls}: {kl_loss.item()}, requires_grad: {kl_loss.requires_grad}")
        return kl_loss

    # 计算SSIM损失 元学习中元损失
    def compute_ssim_loss(self, malicious_decoder, benign_decoder, cls):
        # 从恶意decoder和良性decoder中获取该类别生成的数据
        malicious_generated_data, malicious_labels = self.generate_data_from_decoders([malicious_decoder],
                                                                  self.classifier_num_train_samples)
        benign_generated_data, benign_labels = self.generate_data_from_decoders([benign_decoder],
                                                                  self.classifier_num_train_samples)
        malicious_data_for_class = []
        for i in range(malicious_generated_data.size(0)):
            category = malicious_labels[i].argmax().item()
            if category == cls:  # 只选择类别为cls的数据
                malicious_data_for_class.append(malicious_generated_data[i])

        benign_data_for_class = []
        for i in range(benign_generated_data.size(0)):
            category = benign_labels[i].argmax().item()
            if category == cls:  # 只选择类别为cls的数据
                benign_data_for_class.append(benign_generated_data[i])

        dataset = self.base_params.get("dataset_name")
        category_losses = []
        for m_d in malicious_data_for_class:
            for b_d in benign_data_for_class:
                m_d_np = m_d.detach().cpu().numpy()  # 确保 tensor 在 CPU 上并转换为 NumPy 数组
                b_d_np = b_d.detach().cpu().numpy()
                if dataset in ['mnist', 'fmnist']:
                    ssim_value = ssim(m_d_np.squeeze(),
                                      b_d_np.squeeze(),
                                      data_range=1, multichannel=False)
                elif dataset == 'svhn':  # 三通道彩色图像
                    ssim_value = ssim(m_d_np.transpose(1, 2, 0),
                                      b_d_np.transpose(1, 2, 0),
                                      data_range=1, multichannel=True)
                else:
                    raise ValueError("dataset not in [mnist,fmnist,svhn]")
                ssim_loss = 1 - ssim_value  # 使用1 - SSIM
                category_losses.append(ssim_loss)  # 记录与当前样本相同类别的所有loss
        if category_losses:
            average_loss = torch.mean(torch.tensor(category_losses, device=self.device, requires_grad=True))
        else:
            average_loss = torch.zeros(1, device=self.device, requires_grad=True)
        print(f"SSIM Loss for class {cls}: {average_loss.item()}, requires_grad: {average_loss.requires_grad}")
        return average_loss.clone().detach().requires_grad_(True).to(self.device)

# 生成IID数据集
    def generate_iid_data(self, users, benign_decoders_membership, num_classes):
        # 初始化存储每个类别的生成数据数量
        data_per_class = {cls: 0 for cls in range(num_classes)}
        total_samples_per_class = self.classifier_num_train_samples / num_classes
        print("total_samples_per_class:", total_samples_per_class)
        # 初始化每个类别下每个decoder生成的比例
        # decoder_proportions = {cls: [] for cls in range(num_classes)}
        # decoder_proportions =[]
        X_vals = torch.Tensor()
        y_vals = torch.Tensor()
        # 遍历每个类别
        for cls in range(num_classes):
            # 获取所有可以生成该类别的decoders索引
            decoders_for_class = [decoder for decoder in range(len(users)) if users[decoder].pmf[cls] > 0]
            print(f"decoders_for_class {cls}:{decoders_for_class}")
            # 获取每个decoder的隶属度和训练数据量
            membership_vals = np.array([benign_decoders_membership[decoder] for decoder in decoders_for_class])
            print(f"membership_vals {cls}:{membership_vals}")
            train_data_vals = []
            for decoder in decoders_for_class:
                count_dict = self.compute_data_amt(decoder)
                cls_num = count_dict.get(cls)
                train_data_vals.append(cls_num)
            print(f"train_data_vals {cls}:{train_data_vals}")
            # 1. 隶属度归一化
            normalized_membership = membership_vals / np.sum(membership_vals)
            # 2. 训练数据量归一化
            normalized_train_data = train_data_vals / np.sum(train_data_vals)
            # 3. 隶属度和训练数据量的乘积
            product_vals = normalized_membership * normalized_train_data
            # 4. 对乘积结果归一化
            final_proportions = product_vals / np.sum(product_vals)
            print(f"final_proportions {cls}:{final_proportions}")

            valid_indices = [i for i, size in enumerate((int(total_samples_per_class * p) for p in final_proportions))
                             if size > 0]
            final_proportions = final_proportions[valid_indices]
            final_proportions = final_proportions / final_proportions.sum()  # 归一化
            decoders_for_class = np.array(decoders_for_class)
            decoders_for_class = decoders_for_class[valid_indices]
            # 生成每个类别的数据
            for index, u_id in enumerate(decoders_for_class):
                # if int(total_samples_per_class * final_proportions[index]) > 0:
                users[u_id].model.eval()
                z = users[u_id].model.sample_z(
                    int(total_samples_per_class * final_proportions[index]), "truncnorm", width=self.uniform_range
                ).to(self.device)
                y = torch.full((int(total_samples_per_class * final_proportions[index]),), cls, dtype=torch.long)

                y_hot = one_hot_encode(y, num_classes).to(self.device)
                _, images = users[u_id].model.decoder(z, y_hot)
                X = images.detach()
                X, y_hot = X.cpu(), y_hot.cpu()
                X_vals = torch.cat((X_vals, X), 0)
                y_vals = torch.cat((y_vals, y_hot), 0)
                # else:
                #     print(f"解码器 {u_id} 为类别 {cls} 生成的样本数量为零，已跳过。")
        X_vals = torch.sigmoid(X_vals)
        decoder_dataset = WrapperDatasetG(X_vals, y_vals)
        dl = DataLoader(decoder_dataset, shuffle=True, batch_size=32)
        print(f"分类器训练数据生成完毕,每个类数据量相同,共生成{len(decoder_dataset)}个数据")
        return dl  # 返回生成的IID数据

# 用于IID数据集生成函数，获得类别训练数据量
    def compute_data_amt(self, u):
        """
        Helper function to get probabilities for user target values

        :param u: User ID
        :return: Probability distribution of target data for a given user
        """

        targets = torch.from_numpy(
            np.array(
                [
                    int(self.data_subsets[u][i][1])
                    for i in range(len(self.data_subsets[u]))
                ]
            )
        )

        vals, counts = torch.unique(targets, return_counts=True)  # 返回两个tensor:每个独立元素，以及每个元素的个数
        # print("taarget:",targets)
        # print("######vals:",vals,";##counts:",counts)
        count_dict = {}
        for i in range(len(vals)):
            count_dict[int(vals[i])] = int(counts[i])

        return count_dict

    def train(self):
        # Ensure all models are initialized the same
        if self.should_initialize_same:
            weight_init_state_dict = self.users[0].model.state_dict()
            for u in self.users:
                u.model.load_state_dict(copy.deepcopy(weight_init_state_dict))

        self.evaluate(0)

        selected_users = self.sample_users()

        # Train selected users and collect their decoder weights
        for u in selected_users:
            u.train(self.local_epochs)

        print(f"Finished training user models for epoch 0")
    # 计算每个decoder的三个度量值,并归一化
        # 计算wasserstein距离
        wasserstein_scores = self.calculate_wasserstein_scores(selected_users)
        normalized_wasserstein = self.min_max_normalize(wasserstein_scores)
        print(f"Finished computing wasserstein distance for epoch 0")
        # 计算cosine距离
        cosine_scores = self.calculate_cosine_scores(selected_users)
        normalized_cosine = self.min_max_normalize(cosine_scores)
        print(f"Finished computing cosine distance for epoch 0")
        # 计算ssim距离
        dataset = self.base_params.get("dataset_name")
        ssim_scores = self.calculate_ssim_scores(selected_users, self.classifier_num_train_samples, dataset)
        normalized_ssim = self.min_max_normalize(ssim_scores)
        print(f"Finished computing ssim similarity for epoch 0")
    # 特征向量
        combined_lists = list(zip(normalized_wasserstein, normalized_cosine, normalized_ssim))
        print('特征矩阵combined_lists:', combined_lists)
        # 打印新的组合列表，例子combined_lists：[(0.2, 0.1, 1.0), (0.5, 0.4, 0.8), (0.3, 0.2, 0.9), (0.7, 0.8, 0.7)]
        feature_vectors = np.array(combined_lists)
    # FCM模糊聚类设置
        num_clusters = 2  # 假设我们聚类为2个簇：良性簇和恶意簇
        fuzziness = 2.0   # 设置模糊因子
    # 使用skfuzzy库的cmeans方法进行FCM聚类
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            feature_vectors.T,  # 注意特征矩阵需要转置 (每行是一个特征，列是不同decoder的值)
            num_clusters,  # 聚类的数量
            fuzziness,  # 模糊因子m
            error=0.005,  # 误差阈值
            maxiter=1000,  # 最大迭代次数
            init=None  # 初始隶属度矩阵
        )
        # u是隶属度矩阵，大小为 (num_clusters, num_decoders)
        # u[i][j] 表示第j个decoder在第i个簇中的隶属度
        # print("隶属度矩阵: \n", u)
    # 根据聚类中心分辨良心和恶意簇
        # 对每个聚类中心的特征向量求和，特征值越小的簇是良性簇
        cluster_scores = np.sum(cntr, axis=1)
        # 找到分数最低的簇作为良性簇，较高的簇作为恶意簇
        benign_cluster_idx = np.argmin(cluster_scores)
        malicious_cluster_idx = 1 - benign_cluster_idx
        # 初始化三个列表，用于记录属于良性簇和恶意簇、模糊簇的decoder的索引
        benign_decoders = []
        malicious_decoders = []
        uncertain_decoders = []
        benign_decoders_membership = []
        # 输出每个decoder的隶属度
        for i in range(len(selected_users)):
            benign_membership = u[benign_cluster_idx][i]
            malicious_membership = u[malicious_cluster_idx][i]
            benign_decoders_membership.append(benign_membership)
            # print(f"Decoder {i } 隶属度 - 良性簇: {benign_membership:.4f}, 恶意簇: {malicious_membership:.4f}")
        # 根据隶属度判断是否属于良性簇、恶意簇、模糊簇
        for i in range(len(selected_users)):
            benign_membership = u[benign_cluster_idx][i]
            malicious_membership = u[malicious_cluster_idx][i]
            if benign_membership > self.fuzzy_v:
                benign_decoders.append(i)
                # print(f"Decoder {i} 被归类为良性")  #索引从0开始
            elif malicious_membership > self.fuzzy_v:
                malicious_decoders.append(i)
                # print(f"Decoder {i} 被归类为恶意")
            else:
                uncertain_decoders.append(i)
                # print(f"Decoder {i} 被归类为模糊")
        print(f"良性簇: {benign_decoders}, 恶意簇: {malicious_decoders},模糊簇：{uncertain_decoders}")
    # 元学习：利用良性簇纠正恶意和模糊簇的decoders
        self.correct_malicious_decoders(selected_users, benign_decoders, malicious_decoders, uncertain_decoders,
                                        self.meta_beta, maml_lr=0.01, outer_steps=1)
        print(f"Finished meta-learning correction for epoch 0")
    # Generate a dataloader holding the generated images and labels
        self.classifier_dataloader = self.generate_iid_data(selected_users, benign_decoders_membership, self.num_classes)

        # Train the server model's classifier
        self.train_classifier(reinitialize_weights=True)
        print(f"Trained server classifier for epoch 0")

        print("__________________________________________")

        # Qualitative image check - misc user!
        # self.qualitative_check(
        #     1, self.users[0].model.decoder, "Novel images user 0 decoder"
        # )
