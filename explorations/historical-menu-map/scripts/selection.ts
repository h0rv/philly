export const nyplMenuIds = new Set([
  13759, 14050, 14110, 14406, 21050, 21244, 21577, 21582, 21701, 21754, 21869, 21962, 22008, 22280,
  22295, 22396, 22416, 22467, 22519, 22577, 22759, 22779, 22780, 22882, 22976, 23128, 23209, 23277,
  23323, 23436, 23473, 23612, 23721, 23775, 23995, 24054, 24074, 24475, 24606, 24819, 24827, 25348,
  25574, 25708, 25775, 25857, 25999, 26051, 26300, 27260, 27457, 28620, 29428, 29916, 29996, 30086,
  30411, 30431, 30462, 30472, 30777, 32047, 32212, 33270, 33548, 33662, 33793, 33796, 33815, 33972,
  34327, 34841,
]);

export const nyplArchiveUrl =
  "https://s3.amazonaws.com/menusdata.nypl.org/gzips/2021_08_01_07_01_17_data.tgz";
export const ciaQueryUrl =
  "https://cdm16940.contentdm.oclc.org/digital/bl/dmwebservices/index.php?q=dmQuery/p16940coll1/locati^Philadelphia^exact^and/title!date!descri!decade!locati/title/1024/1/0/0/json";
export const ciaApiUrl = (id: string): string =>
  `https://cdm16940.contentdm.oclc.org/digital/api/singleitem/collection/p16940coll1/id/${id}`;
